# -*- coding: utf-8 -*-
from core.agents import Analyst, Planner, Inspector, Designer
from core.prompts import MAX_ROUND, SYSTEM_NAME, ANALYST_NAME, PLANNER_NAME, INSPECTOR_NAME

LLM_API_FUC = None
try:
    from core import api
    LLM_API_FUC = api.safe_call_llm
    print(f"Use func from core.api in chat_manager.py")
except:
    from core import llm
    LLM_API_FUC = llm.safe_call_llm
    print(f"Use func from core.llm in chat_manager.py")

import time
from pprint import pprint


class ChatManager(object):
    def __init__(self, ontology_json_path: str, log_path: str, model_name: str, kg_id:str, lazy: bool=False, without_selector: bool=False):
        self.ontology_json_path = ontology_json_path # path to table description json file
        self.log_path = log_path  # path to record important printed content during running
        self.model_name = model_name  # name of base LLM called by agent
        self.dataset_name = kg_id
        self.ping_network()
        self.chat_group = [
            Analyst(ontology_json_path=self.ontology_json_path, model_name=self.model_name, kg_id=kg_id, lazy=lazy, without_selector=without_selector),
            Designer(kg_id=kg_id),
            Planner(kg_id=kg_id),
            Inspector(ontology_json_path=self.ontology_json_path, kg_id=kg_id)
        ]

    def ping_network(self):
        # check network status
        print("Checking network status...", flush=True)
        try:
            _ = LLM_API_FUC("Hello world!")
            print("Network is available", flush=True)
        except Exception as e:
            raise Exception(f"Network is not available: {e}")

    def _chat_single_round(self, message: dict):
        for agent in self.chat_group:  # check each agent in the group
            if message['send_to'] == agent.name:
                agent.talk(message)

    def start(self, user_message: dict):
        # we use `dict` type so value can be changed in the function
        start_time = time.time()
        if user_message['send_to'] == SYSTEM_NAME:  # in the first round, pass message to prune
            user_message['send_to'] = ANALYST_NAME
        for _ in range(MAX_ROUND):  # start chat in group
            self._chat_single_round(user_message)
            if user_message['send_to'] == SYSTEM_NAME:  # should terminate chat
                break
        end_time = time.time()
        exec_time = end_time - start_time
        print(f"\033[0;34mExecute {exec_time} seconds\033[0m", flush=True)
