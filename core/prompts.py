
from langchain_core.prompts import PromptTemplate

MAX_ROUND = 3  # max try times of one agent talk

ANALYST_NAME = 'Analyst'
PLANNER_NAME = 'Planner'
INSPECTOR_NAME = 'Inspector'
DESIGNER_NAME = 'Designer'
SYSTEM_NAME = 'System'


analyst_template = """
[Instruction]
As a professional knowledge engineer, your task is to extract schema items from [Knowledge graph schema] relevant to answer [Question]. Please adhere to the following guidelines:

1. Discard any schema items that is not related to the question.
2. Sort the concepts and relations in descending order.
3. Ensure at least 1 relation and 1 concept included in the final output. 
4. The output should be in JSON format.

Here are some examples:

==========
【Knowledge base schema】
# Concepts:
Film, Channel, Organisation, Company, Artist, Type, Place, City, Industry, Occupation, Person, Country, Background, Service, Cinematography, Language, Station, Club, University
# Relations:
timeshiftChannel, director, imdbId, distributor, musicComposer, type, deathDate, club, editing, releaseDate, child, producer, birthPlace, headquarter, deathPlace, located, budget, runtime, owner, starring, iso6392Code, writer, industry, activeYearsStartYear, occupation, sisterStation, spouse, spokenIn, gross, formerName, birthYear, birthName, editor, language, background, service, cinematography, deathYear, birthDate, keyPerson, founders, iso6391Code, foundingYear, broadcastedBy 
【Question】
Who was born in 1985 among the wives of Meta Platforms' co-founders?
【Answer】
```json
{{
  "concepts": ["Company", "Person"],
  "relations": ["founders", "spouse", "birthDate"]
}}
```
Question Solved.

==========

Here is a new example, please start answering:

[Knowledge base schema]
{desc_str}
[Question]
{query}
[Answer]
"""


planner_template = """
[Instruction]
As a professional knowledge engineer, your task is to generate a SPARQL query for the [Question] by filling up the [Template] with the actual values. You need to decompose the [Question] into sub-questions, and fill in the corresponding values in [Template] of each sub-questions. Please adhere to the following guidelines:

1. A special variable ´?ans´is used by default. If multiple intents are detected, use separate variables like ´?ans1´, ´?ans2´, and so on for each result.
2. Priority is given to utilizing the information provided in [Knowledge graph schema].
3. Excluded all prefixes.
4. The output should be in JSON format.

Here are some examples:

[Knowledge graph schema]
# Concepts: 
Company, Person
# Relations: 
spouse (Person, Person), founders (Company, Person), birthDate (Person, Literal)
# Entities:
Meta_Platforms
[Question]
Who was born in 1985 among the wives of Meta Platforms’ co-founders?
[Template]
´´´template
SELECT ?ans WHERE  {{
    [ent] [rel1] [var1] .
    [var1] [rel2] [var2] .
    [var2] [rel3] ?ans .
    FILTER [con]
}}
´´´

Decompose the question into sub questions, and generate the SPARQL query after thinking step by step:
Sub question 1: Who are Meta Platforms' co-founders?
```sparql
SELECT ?ans WHERE {{ 
    Meta_Platforms founders ?var1 .
    [var1] [rel2] [var2] .
    [var2] [rel3] ?ans .
    FILTER [con]
}}
```

Sub question 2: Who are the wives of Meta Platforms' co-founders?
```sparql
SELECT ?ans WHERE {{ 
    Meta_Platforms founders ?var1 .
    ?var1 spouse ?var2 .
    [var2] [rel3] ?ans .
    FILTER [con]
}}
```

Sub question 3: When were the wives of Meta Platforms' co-founders born?
```sparql
SELECT ?ans WHERE {{ 
    Meta_Platforms founders ?var1 .
    ?var1 spouse ?var2 .
    ?var2 birthDate ?ans .
    FILTER [con]
}}
```

Sub question 4: Who was born in 1985 among the wives of Meta Platforms’ co-founders?
```sparql
SELECT DISTINCT ?ans WHERE {{ 
    Meta_Platforms founders ?var1 .
    ?var1 spouse ?var2 .
    ?var2 birthDate ?ans .
    FILTER ( year ( ?ans ) = 1985 )
}}
```

Question Solved.

Here is a new example:

[Knowledge base schema]
 {desc_str}
[Question]
 {query}
 [Template]
 {template}

Decompose the question into sub questions, and generate the SPARQL template after thinking step by step:

"""


designer_template = """
[Instruction]
As a professional knowledge engineer, your task is to generate a SPARQL template for the [Question] according to the [Generic SPARQL Template] and [Knowledge graph schema]. You need to first identify the question type and refers to the basic example template of that type, and refines it based on the question and knowledge graph schema. Please adhere to the following guidelines:

1. When multiple variables are required, use distinct placeholders such as [var1], [var2], etc. Similarly, use [ent1], [ent2], etc., for entities and [rel1], [rel2], etc., for relations.
2. The template is restricted to only include predefined placeholders and reserved keywords of SPARQL queries; all other elements are excluded.
3. The subject and object of the relation placeholder [rel] in each triple pattern must conform to the domain and range specifications defined in [Knowledge graph schema].
4. The number of required triple patterns should correspond to the reasoning path implied by the [Question] and the [Knowledge graph schema].
5. the gold entities are not provided in [Knowledge graph schema], add an extra triple ´[var] rdfs:label [val]´ for label matching or constraint ´FILTER [con]´ for string or regex matching. 
6. The output should be in JSON format.

[Generic SPARQL Template]
A Generic SPARQL Template consists of the following three types of elements: 
 - Reserved Keywords – Standard SPARQL operators such as SELECT, ASK, FILTER, COUNT, etc.
 - Literals – Numerical values, strings, dates, and other fixed textual elements.
 - KG-specific Identifiers – Entities, relations, concepts, and variables derived from the knowledge graph (KG).
Six special tokens are used as placeholders for different SPARQL components:
 - [ent] for entities mentioned in the query,
 - [cct] for concepts mentioned in the query,
 - [rel] for relations defined in the KG ontology,
 - [var] for variables,
 - [val] for literals used in graph patterns, value clauses, or solution modifiers,
 - [con] for constraints in SPARQL clauses (FILTER, ORDER BY, GROUP BY, HAVING, LIMIT).
 
 
 [Question Types]
  - Type 1: Simple questions
[Question]
 Who founded Meta Platforms?
[Template]
´´´template
SELECT ?ans WHERE { 
[ent] [rel] ?ans . 
}
´´´

 - Type 2: Simple questions with aggregator(s)
[Question]
How many co-founders does Meta Platforms have?
[Template]
´´´template
SELECT ( COUNT ( [var] ) AS ?ans ) WHERE {{
    [ent] [rel] ?ans .
}}
´´´

 - Type 3: Multi-hop
[Question]
List all wives of Meta Platforms' co-founders?
[Template]
´´´template
SELECT ?ans WHERE { 
[ent] [rel1] [var1] . 
[var1] [rel2] ?ans . 
}
´´´

 - Type 4: Multi-hop with constraint(s)
 [Question]
Who was born in 1984 among Meta Platforms’ co-founders?
【Template】
´´´template
SELECT ?ans WHERE { 
[ent] [rel1] [var1] . 
[var1] [rel2] ?ans . 
FILTER [con] 
}
´´´

 - Type 5: Multi-hop with aggregators
[Question]
What is the largest among all the birth dates of Meta Platforms’ co-founders?
[Template]
´´´template
SELECT ( MIN ( [var] ) as ?ans ) WHERE { 
[ent] [rel1] [var1] . 
[var1] [rel2] ?ans . 
} 
´´´

 - Type 5: Multi-hop with aggregator(s) and constraint(s)
[Question]
What is the average age of Meta Platforms' co-founders who were born after 1980?
[Template]
´´´template
SELECT ( COUNT ( [var] ) as ?ans ) WHERE { 
[ent] [rel1] [var1] . 
[var1] [rel2] ?ans . 
FILTER [con] 
}
´´´ 

Here are some examples:
[Knowledge graph schema]
# Concepts: 
Company, Person
# Relations: 
spouse (Person, Person), founders (Company, Person), birthDate (Person, Literal)
# Entities:
Meta_Platforms
[Question]
Who was born in 1985 among the wives of Meta Platforms’ co-founders?
[Template]
´´´template
SELECT ?ans WHERE  {{
    [ent] [rel1] [var1] .
    [var1] [rel2] [var2] .
    [var2] [rel3] ?ans .
    FILTER [con]
}}

Here is a new example, please think step by step:

[Knowledge base schema]
{desc_str}
[Question]
{query}
[Template]

"""


inspector_template = """
[Instruction]
As a professional knowledge engineer, your task is to inspect and refine the [Old SPARQL] for the [Question] based on [Knowledge graph schema]. If [SPARQL Error] and [Exception Class] occur, please fix it up and output the final [Correct SPARQL]. 

[Knowledge base schema]
{desc_str}
[Question]
{query}
[Old SPARQL]
{old_sparql}
[SPARQL error]
{sparql_error}
[Exception class]
{exception_class}

Now, please start the process.
[Correct SPARQL]

"""

