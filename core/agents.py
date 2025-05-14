# -*- coding: utf-8 -*-
from core.utils import parse_json, parse_sparql_from_string, parse_template_from_string, load_json_file, extract_world_info, postprocess
from func_timeout import func_set_timeout, FunctionTimedOut
from typing import List, Tuple
from SPARQLWrapper import SPARQLWrapper, JSON
import json
import urllib

sparql = SPARQLWrapper("http://localhost:8890/sparql")
sparql.setReturnFormat(JSON)

LLM_API_FUC = None
try:
    from core import api
    LLM_API_FUC = api.safe_call_llm
    print(f"Use func from core.api in agents.py")
except:
    from core import llm
    LLM_API_FUC = llm.safe_call_llm
    print(f"Use func from core.llm in agents.py")

from core.const import *
from typing import List
from copy import deepcopy

import time
import abc
import sys
import os
from tqdm import tqdm, trange


class BaseAgent(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def talk(self, message: dict):
        pass


class Analyst(BaseAgent):
    name = ANALYST_NAME
    description = "Get knowledge base description and if need, extract relative concepts & relations"

    def __init__(self, ontology_json_path: str, model_name: str, kg_id:str, lazy: bool = False, without_selector: bool = False):
        super().__init__()
        self.ontology_json_path = ontology_json_path
        self.model_name = model_name
        self.kg_id = kg_id
        self.ontology = {}
        self.init_kb()
        self._message = {}
        self.without_selector = without_selector
    
    def init_kb(self):
        if not os.path.exists(self.ontology_json_path):
            raise FileNotFoundError(f"ontology.json not found in {self.ontology_json_path}")
        self.ontology = load_json_file(self.ontology_json_path)

    def _build_kb_schema_list_str(self, ontology, extracted_concepts, extracted_relations, verbose=True):
        schema_desc_str = ''
        if verbose:
            concepts = ontology["classes"]
            relations = ontology["relations"]
            schema_desc_str += "# Concepts: \n"
            concept_str = f"{{clsname}}: {{description}}"
            schema_desc_str += "\n".join([concept_str.format(clsname=k, description=v["description"]) for k, v in concepts.items() if k in extracted_concepts])
            schema_desc_str += "\n# Relations: \n"
            tmp_relations = []
            for rel in relations:
                for ext_rel in extracted_relations:
                    if ext_rel in rel:
                        tmp_relations.append(rel)
            rel_str = f"{{relname}} ({{domain}}, {{range}}) "
            schema_desc_str += ", ".join([rel_str.format(relname=k,domain=v["domain"],range=v["range"]) for k, v in relations.items() if k in tmp_relations])
        else:
            schema_desc_str = ''
            schema_desc_str += "# Concepts: \n"
            schema_desc_str += ", ".join([concept for concept in extracted_concepts])
            schema_desc_str += "\n# Relations: \n"
            schema_desc_str += ", ".join([relation for relation in extracted_relations])
        return schema_desc_str

    
    def _get_kb_desc_str(self,
                         extracted_schema: dict,
                         use_gold_schema: bool = False):

        concept_info = self.ontology['classes']  # concept:str -> concepts[(concept_name, full_concept_name, extra_concept_desc): str]
        relation_info = self.ontology['relations']  # relation:str -> relations[(relation_name, full_relation_name): str]
        entity_info = self.ontology['entities']

        schema_desc_str = ''  # for concat

        chosen_db_schem_dict = {} # {"concepts": [], "relations": []}

        if len(extracted_schema) == 0 or len(extracted_schema["concepts"]) == 0 and len(extracted_schema["relations"]) == 0:
            extracted_schema["concepts"] = [k for k, v in concept_info.items()]
            extracted_schema["relations"] = [k for k,v in relation_info.items()]

        chosen_db_schem_dict["concepts"] = extracted_schema["concepts"]
        chosen_db_schem_dict["relations"] = extracted_schema["relations"]
        chosen_db_schem_dict["entities"] = entity_info

        schema_desc_str += self._build_kb_schema_list_str(self.ontology, extracted_schema["concepts"], extracted_schema["relations"])

        schema_desc_str = schema_desc_str.strip()
        
        return schema_desc_str, chosen_db_schem_dict

    def _is_need_prune(self, db_schema: str):
        if len(self.ontology["classes"]) <= 10 or len(self.ontology["relations"]) <= 10:
            return False
        else:
            return True

    def _prune(self,
               query: str,
               db_schema: str
               ) -> dict:
        prompt = analyst_template.format(query=query, desc_str=db_schema)
        word_info = extract_world_info(self._message)
        reply = LLM_API_FUC(prompt, **word_info)
        extracted_schema_dict = parse_json(reply)
        return extracted_schema_dict

    def talk(self, message: dict):
        if message['send_to'] != self.name: return
        self._message = message
        ext_sch, question = message.get('extracted_schema', {}), message.get('question')
        use_gold_schema = False
        if ext_sch: # extracted schema
            use_gold_schema = True
        db_schema, chosen_db_schem_dict = self._get_kb_desc_str(extracted_schema=ext_sch, use_gold_schema=use_gold_schema)
        need_prune = self._is_need_prune(db_schema)
        if self.without_selector:
            need_prune = False
        if need_prune:
            
            try:
                raw_extracted_schema_dict = self._prune(query=question, db_schema=db_schema)
            except Exception as e:
                print(e)
                raw_extracted_schema_dict = {}
            
            print(f"query: {message['question']}\n")
            db_schema_str, chosen_db_schem_dict = self._get_kb_desc_str(extracted_schema=raw_extracted_schema_dict)

            message['extracted_schema'] = raw_extracted_schema_dict
            message['chosen_db_schem_dict'] = chosen_db_schem_dict
            message['desc_str'] = db_schema_str
            message['pruned'] = True
            message['send_to'] = DESIGNER_NAME
        else:
            message['chosen_db_schem_dict'] = chosen_db_schem_dict
            message['desc_str'] = db_schema
            message['pruned'] = False
            message['send_to'] = DESIGNER_NAME


class Planner(BaseAgent):
    """
    Decompose the question and solve them using CoT
    """
    name = PLANNER_NAME
    description = "Decompose the question and solve them using CoT"

    def __init__(self, kg_id: str):
        super().__init__()
        self.kg_id = kg_id
        self._message = {}

    def talk(self, message: dict):
        if message['send_to'] != self.name: return
        self._message = message
        query, schema_info, template, entities = message.get('question'), message.get('desc_str'), message.get('template'), message.get("entities")

        chosen_db_schem_dict = message.get("chosen_db_schem_dict")
        entity_info = chosen_db_schem_dict["entities"]
        schema_info += "\n# Entities: \n"
        for ent_info in entity_info:
            for ent in entities:
                if ent in ent_info["qid"]:
                    schema_info += ent_info["qid"] + " (" + ent_info["label"] + "), "

        prompt = planner_template.format(desc_str=schema_info, query=query, template=template)

        word_info = extract_world_info(self._message)
        reply = LLM_API_FUC(prompt, **word_info).strip()
        
        res = ''
        qa_pairs = reply
        
        try:
            res = parse_sparql_from_string(reply)
        except Exception as e:
            res = f'error: {str(e)}'
            print(res)
            time.sleep(1)
        
        message['final_sparql'] = res
        message['qa_pairs'] = qa_pairs
        message['fixed'] = False
        message["desc_str"] = schema_info
        message['send_to'] = INSPECTOR_NAME


class Designer(BaseAgent):
    name = DESIGNER_NAME
    description = "Predict SPARQL query template"

    def __init__(self, kg_id: str):
        super().__init__()
        self.kg_id = kg_id
        self._message = {}

    @func_set_timeout(120)
    def _predict_template(self, question: str):
        pass

    def talk(self, message: dict):
        """
        Execute SQL and preform validation
        :param message: {"query": user_query,
                        "desc_str": description of db schema,
                        "final_sparql": generated SPARQL to be verified}
        :return: execution result and if need, refine SQL according to error info
        """
        if message['send_to'] != self.name: return
        self._message = message
        query, schema_info, entities = message.get('question'), message.get('desc_str'), message.get("entities")

        chosen_db_schem_dict = message.get("chosen_db_schem_dict")
        entity_info = chosen_db_schem_dict["entities"]
        schema_info += "\n# Entities: \n"
        for ent_info in entity_info:
            for ent in entities:
                if ent in ent_info["qid"]:
                    schema_info += ent_info["qid"] + " (" + ent_info["label"] + "), "

        prompt = designer_template.format(query=query, desc_str=schema_info)

        word_info = extract_world_info(self._message)
        reply = LLM_API_FUC(prompt, **word_info).strip()

        res = ''
        qa_pairs = reply

        try:
            res = parse_template_from_string(reply)
        except Exception as e:
            res = f'error: {str(e)}'
            print(res)
            time.sleep(1)

        message['template'] = res
        message['qa_pairs'] = qa_pairs
        message['fixed'] = False
        message['send_to'] = PLANNER_NAME


class Inspector(BaseAgent):
    name = INSPECTOR_NAME
    description = "Execute SPARQL and preform validation"

    def __init__(self, ontology_json_path:str, kg_id: str):
        super().__init__()
        self.ontology_json_path = ontology_json_path
        self.kg_id = kg_id
        self._message = {}
        self.init_kb()

    def init_kb(self):
        if not os.path.exists(self.ontology_json_path):
            raise FileNotFoundError(f"ontology.json not found in {self.ontology_json_path}")
        self.ontology = load_json_file(self.ontology_json_path)

    @func_set_timeout(120)
    def _execute_sparql(self, query: str) -> dict:

        sparql.setQuery(query)
        try:
            response = sparql.query().convert()
            rtn = []
            if "boolean" in response:  # ASK
                rtn = [response["boolean"]]
            else:
                if len(response["results"]["bindings"]) > 0 and "callret-0" in response["results"]["bindings"][
                    0]:  # COUNT
                    rtn = [int(response['results']['bindings'][0]['callret-0']['value'])]
                else:
                    for res in response['results']['bindings']:
                        res = {k: v["value"] for k, v in res.items()}
                        rtn.append(res)

            return {
                "sparql": str(query),
                "data": rtn,
                "sparql_error": "",
                "exception_class": ""
            }
        except urllib.error.URLError as er:
            return {
                "sparql": str(query),
                "sparql_error": str(' '.join(er.args)),
                "exception_class": str(er.__class__)
            }

    def _is_need_refine(self, exec_result: dict):
        
        data = exec_result.get('data', None)
        if data is not None:
            if len(data) == 0:
                exec_result['sparql_error'] = 'no data selected.'
                return True
            return False
        else:
            return True

    def _refine(self,
               query: str,
               schema_info: str,
               error_info: dict) -> dict:

        sparql = error_info.get('sparql')
        sparql_error = error_info.get('sparql_error')
        exception_class = error_info.get('exception_class')
        prompt = inspector_template.format(query=query, sparql=sparql, sparql_error=sparql_error, desc_str=schema_info, exception_class=exception_class)

        word_info = extract_world_info(self._message)
        reply = LLM_API_FUC(prompt, **word_info)
        res = parse_sparql_from_string(reply)
        return res

    def talk(self, message: dict):
        if message['send_to'] != self.name: return
        self._message = message
        old_sparql, query, schema_info, kg_id = message.get('pred', message.get('final_sparql')), \
                                                            message.get('question'), \
                                                            message.get('desc_str'), \
                                                            message.get("kg_id")
        old_sparql = postprocess(old_sparql, kg_id, self.ontology)

        if 'error' in old_sparql:
            message['try_times'] = message.get('try_times', 0) + 1
            message['pred'] = old_sparql
            message['send_to'] = SYSTEM_NAME
            return
        
        is_timeout = False
        is_need = False
        try:
            error_info = self._execute_sparql(old_sparql)
            is_need = self._is_need_refine(error_info)
        except Exception as e:
            is_timeout = True
        except FunctionTimedOut as fto:
            is_timeout = True

        # is_need = False
        if not is_need or is_timeout:  # correct in one pass or refine success or timeout
            message['try_times'] = message.get('try_times', 0) + 1
            message['pred'] = old_sparql
            message['send_to'] = SYSTEM_NAME
        else:
            new_sparql = self._refine(query, schema_info, error_info)
            new_sparql = postprocess(new_sparql, kg_id, self.ontology)
            message['try_times'] = message.get('try_times', 0) + 1
            message['pred'] = new_sparql
            message['fixed'] = True
            message['send_to'] = INSPECTOR_NAME
        return