import pickle
import pandas as pd
import sys
import numpy as np
import networkx as nx
# from PC_Causal import LLMClient
from LLM_Causal_Graph_new import CausalGraph
np.random.seed(211)
import pandas as pd
import bnlearn as bn
from pycit import citest
from LLM_Causal_Graph_new import CausalGraph
from pgmpy.utils import get_example_model
from pgmpy.estimators import PC
import re
from tqdm import tqdm
import pandas as pd
import numpy as np

from LLM_Causal_Graph_new import CausalGraph
# from data_handle import BnNetwork
np.random.seed(211)
import pandas as pd
# from data_handle import BnNetwork
# from scipy.stats import chi2_contingency
import openai
from LLM_Causal_Graph_new import CausalGraph
import re
from tqdm import tqdm
import time

import itertools
import traceback
import time

import openai
from openai import OpenAI
import ast
import os

import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
import sys
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import networkx as nx
from LLM_Causal_Graph_new import CausalGraph

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")



pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}


SYSTEM = "You are a helpful assistant on causal reasoning. Your goal is to answer factually and concisely questions about cause and effect."
NUM_EVALS = None#None # 50
NUM_TRIES = 1
SKIP_PROMPTS = 0
CHATMODELS = ['gpt-4', 'gpt-3.5-turbo']
AZURE = True
DELAY = 1
PROMPTS = "prompts"
prompt_suffix = "_" + PROMPTS.split("_")[1] if "_" in PROMPTS else ""
# openai.api_type = "azure"
# openai.api_version = "2023-03-15-preview"
# openai.api_base = "https://msri-openai-ifaq.azure-api.net"
# # openai.api_base = "https://gcrgpt4aoai6.openai.azure.com/"
# # openai.api_key = "7afb37c66cea4310a1668151e65443aa"
# openai.api_key = "42a5fa6b1e6c4024be7cecdeb74b6075"


# openai.api_base = "https://gcrendpoint.azurewebsites.net"
# openai.api_version = "2023-05-15"
# openai.api_key = "oaip_uzdjbTvnSYoFUlbpHFlYGusZWdKDsicd"


#Anush Key
# OpenAI.api_key = "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"


client = openai.AzureOpenAI(
    azure_endpoint="https://msri-openai-ifaq.azure-api.net",
    api_key="137bc22159424422a88bf664b287a7c7", #Atharv key
    # api_key="34a50a5d3f3c4bd78877ba5062bc7486", #Lab key
    # api_key="42a5fa6b1e6c4024be7cecdeb74b6075", #Amit key
    api_version="2024-02-15-preview",
    # api_version="2023-12-01-preview",
        )  # noqa



class WeakLearners:
    def subgraph_generator(self, nodes):
        r = 3 # the length of the subgroups

        # using itertools.combinations to generate all subgroups of three elements
        subgroups = list(itertools.combinations(nodes, r))

        # use nested for loops to generate all pairs of elements
        final_pairs = []
        for group in subgroups:
            pairs = []
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    pairs.append((group[i], group[j]))
            final_pairs.append(pairs)

        # print the pairs
        # print(final_pairs)
        return final_pairs
    
    def subgroup_from_nodes(self, nodes):
        r = 3 # the length of the subgroups

        subgroups = list(itertools.combinations(nodes, r))

        return subgroups
    
    def triplets_from_skeleton(self, lst):
        #Given that skeleton is a list of tuples representing an undirected graph, return lists of edges containing pairs of nodes that are connected by a third node
        print(lst)
        pairs = []
        for i in range(len(lst)):
            print(lst[i])
            for j in range(i+1, len(lst)):
                print(lst[j])
                if lst[i][0] == lst[j][0] or lst[i][0] == lst[j][1] or lst[i][1] == lst[j][0] or lst[i][1] == lst[j][1]:
                    pairs.append([lst[i], lst[j]])
        return pairs
    

    def query_gpt4(self, model_name, X, Y, nodes, context):

        input_prompt = f''' 
        Question: For a causal graph used to model relationship of various factors and outcomes related to cancer with the following nodes: ['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea'],
            Which cause-and-effect relationship is more likely between nodes 'smoker' and 'cancer'?

            A. changing the state of node 'smoker' causally effects a change in another node 'cancer'.

            B. changing the state of node 'cancer' causally effects a change in another node 'smoker'.

            C. There is no causal relation between the nodes 'cancer' and 'smoker'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer:  The causal effect of "smoker" directing to "cancer" is based on the strong evidence from epidemiological studies linking smoking to an increased risk of developing cancer. Smoking introduces harmful substances into the respiratory system, leading to cellular damage and mutation, which significantly raises the likelihood of cancer development in the lungs or respiratory tract, subsequently impacting the occurrence of respiratory problems like shortness of breath. Therefore answer is <Answer>A</Answer>

        Question: For a causal graph used to model relationship of various factors and outcomes related to cancer with the following nodes: ['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea'],
            Which cause-and-effect relationship is more likely between nodes 'xray' and 'dyspnoea'?

            A. changing the state of node 'xray' causally effects a change in another node 'dyspnoea'.

            B. changing the state of node 'dyspnoea' causally effects a change in anot~her node 'xray'.

            C. There is no causal relation between the nodes 'xray' and 'dyspnoea'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed. 

        Answer: Reasoning behind the lack of causal relation between X-ray and dyspnoea is that X-ray and dyspnoea are both effects of having cancer, but they do not directly cause or affect each other. X-ray is a diagnostic test that can help detect cancer in the lungs or other organs, while dyspnoea is a symptom of cancer that involves feeling short of breath. Therefore, X-ray and dyspnoea are not causally related, but they are both associated with cancer. Therefore answer is 
        <Answer>C</Answer>

        Question: For a causal graph used to model relationship of various factors and outcomes related to cancer with the following nodes: ['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea'],
            Which cause-and-effect relationship is more likely between nodes 'xray' and 'cancer'?

            A. changing the state of node 'xray' causally effects a change in another node 'cancer'.

            B. changing the state of node 'cancer' causally effects a change in another node 'xray'.

            C. There is no causal relation between the nodes 'xray' and 'cancer'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer:  The causal effect of cancer on X-ray is that X-rays are often used to diagnose or detect cancer in different parts of the body, such as the bones, lungs, breasts, or kidneys123. X-rays use low doses of radiation to create pictures of the inside of the body and show the presence, size, and location of tumors. X-rays can also help monitor the response to treatment or check for signs of recurrence. Therefore, having cancer may increase the likelihood of getting an X-ray as part of the diagnostic process or follow-up care. Therefore answer is <Answer>B</Answer>

        Question: For a causal graph used to model relationship of various factors and outcomes related to cancer with the following nodes: ['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea'],
            Which cause-and-effect relationship is more likely between nodes 'pollution' and 'cancer'?

            A. changing the state of node 'pollution' causally effects a change in another node 'cancer'.

            B. changing the state of node 'cancer' causally effects a change in another node 'pollution'.

            C. There is no causal relation between the nodes 'pollution' and 'cancer'.

           Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer:  The causal effect of pollution on cancer is that air pollution contains carcinogens (cancer-causing substances) that may be absorbed into the body when inhaled and damage the DNA of cells. Another possible reasoning is that particulate matter (tiny dust-like particles) in air pollution may cause physical damage to the cells in the lungs, leading to inflammation and oxidative stress and eventually cell mutations. A third possible reasoning is that air pollution may create an inflamed environment in the lungs that encourages the proliferation of cells with existing cancer-driving mutations. These are some of the hypotheses that researchers have proposed to explain how air pollution may cause cancer, but more studies are needed to confirm them. Therefore answer is <Answer>A</Answer>

        Question: For a causal graph used to model relationship of various factors and outcomes related to cancer with the following nodes: ['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea'],
            Which cause-and-effect relationship is more likely between nodes 'pollution' and 'smoker'?

            A. changing the state of node 'pollution' causally effects a change in another node 'smoker'.

            B. changing the state of node 'smoker' causally effects a change in another node 'pollution'.

            C. There is no causal relation between the nodes 'pollution' and 'smoker'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer: Reason behind the lack of causal relation between pollution and smoker is that pollution and smoking are both independent risk factors for respiratory problems, but they do not directly cause or affect each other. Pollution and smoking both contribute to air pollution, which can harm the health of people and the environment. However, pollution is mainly caused by human activities such as burning fossil fuels, deforestation, or industrial processes, while smoking is a personal choice that involves inhaling tobacco smoke. Therefore, pollution and smoker are not causally related, but they are both associated with respiratory problems. Therefore answer is <Answer>C</Answer>.

        Question: For a causal graph used for modeling factors causing Coronary Heart Diseases with the following nodes: ['Family Disease', 'Gene', 'Smoking', 'Blood Pressure', 'Coronary Heart Disease', 'Headache'],
            Which cause-and-effect relationship is more likely between nodes 'Family Disease' and 'Gene'?

            A. changing the state of node 'Family Disease' causally effects a change in another node 'Gene'.

            B. changing the state of node 'Gene' causally effects a change in another node 'Family Disease'.

            C. There is no causal relation between the nodes 'Family Disease' and 'Gene'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer: Reason behind the causal effect of family disease on gene is that family disease is a term that refers to diseases or health conditions that run in the family, meaning that they are influenced by genetic factors. Gene is a term that refers to the basic unit of heredity that carries information for a specific trait or function. Family disease can affect gene by altering the type or frequency of genes that are inherited by the offspring from their parents. For example, some family diseases are caused by deterministic genes, which are genes that guarantee the development of a disease if they are present in a person’s genome. Other family diseases are influenced by risk genes, which are genes that increase the likelihood of developing a disease but do not guarantee it. Therefore, family disease can causally direct to gene by changing the genetic makeup of a person or a population. Therefore answer is <Answer>A</Answer>.

        Question: For a causal graph used for modeling factors causing Coronary Heart Diseases with the following nodes: ['Family Disease', 'Gene', 'Smoking', 'Blood Pressure', 'Coronary Heart Disease', 'Headache'],
            Which cause-and-effect relationship is more likely between nodes 'Coronary Heart Disease' and 'Gene'?

            A. changing the state of node 'Coronary Heart Disease' causally effects a change in another node 'Gene'.

            B. changing the state of node 'Gene' causally effects a change in another node 'Coronary Heart Disease'.

            C. There is no causal relation between the nodes 'Coronary Heart Disease' and 'Gene'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer: Possible reasoning behind the causal effect of gene on coronary heart disease is that gene is a term that refers to the basic unit of heredity that carries information for a specific trait or function. Gene can affect coronary heart disease by influencing the structure and function of the blood vessels, the metabolism and transport of lipids (fats) in the blood, the inflammation and clotting processes, or the response to environmental factors such as smoking or diet. For example, some genes code for proteins that regulate the cell cycle and growth of the cells that line the arteries, which can affect their susceptibility to damage or plaque formation. Other genes code for proteins that control the synthesis and clearance of cholesterol or other lipids, which can affect their levels and deposition in the arteries. Therefore, gene can causally direct to coronary heart disease by modifying the biological pathways that contribute to the development or progression of the disease. Therefore answer is <Answer>B</Answer>

        Question: For a causal graph used for modeling factors causing Coronary Heart Diseases with the following nodes: ['Family Disease', 'Gene', 'Smoking', 'Blood Pressure', 'Coronary Heart Disease', 'Headache'],
            Which cause-and-effect relationship is more likely between nodes 'Blood Pressure' and 'Smoking'?

            A. changing the state of node 'Blood Pressure' causally effects a change in another node 'Smoking'.

            B. changing the state of node 'Smoking' causally effects a change in another node 'Blood Pressure'.

            C. There is no causal relation between the nodes 'Blood Pressure' and 'Smoking'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer: Possible reasoning behind the causal effect of smoking on blood pressure is that smoking is a habit that involves inhaling tobacco smoke, which contains nicotine and other harmful chemicals. Smoking can affect blood pressure by activating the sympathetic nervous system (SNS), which is the part of the nervous system that controls the body's response to stress or danger. When the SNS is activated, it releases hormones such as adrenaline and noradrenaline, which cause the heart to beat faster and harder, and the blood vessels to constrict. This results in a temporary increase in blood pressure, which can last for 15 to 20 minutes after each cigarette. Therefore, smoking can causally direct to blood pressure by stimulating the SNS and increasing the cardiac output and vascular resistance. Therefore answer is <Answer>B</Answer>.

        Question: For a causal graph used for modeling factors causing Coronary Heart Diseases with the following nodes: ['Family Disease', 'Gene', 'Smoking', 'Blood Pressure', 'Coronary Heart Disease', 'Headache'],
            Which cause-and-effect relationship is more likely between nodes 'Headache' and 'Smoking'?

            A. changing the state of node 'Headache' causally effects a change in another node 'Smoking'.

            B. changing the state of node 'Smoking' causally effects a change in another node 'Headache'.

            C. There is no causal relation between the nodes 'Headache' and 'Smoking'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer: One possible reasoning behind the lack of causal relation between headache and smoking is that headache and smoking are both associated with various health conditions, but they do not directly cause or affect each other12. Headache is a term that refers to pain or discomfort in the head, scalp, or neck, which can have many possible causes, such as stress, dehydration, infection, injury, or medication. Smoking is a habit that involves inhaling tobacco smoke, which contains nicotine and other harmful chemicals, which can increase the risk of diseases such as cancer, heart disease, stroke, and lung disease. Therefore, headache and smoking are not causally related, but they are both linked to different health problems. Therefore the answer is <Answer>C</Answer>

        Question: For a causal graph used for modeling factors causing Coronary Heart Diseases with the following nodes: ['Family Disease', 'Gene', 'Smoking', 'Blood Pressure', 'Coronary Heart Disease', 'Headache'],
            Which cause-and-effect relationship is more likely between nodes 'Headache' and 'Smoking'?

            A. changing the state of node 'Headache' causally effects a change in another node 'Smoking'.

            B. changing the state of node 'Smoking' causally effects a change in another node 'Headache'.

            C. There is no causal relation between the nodes 'Headache' and 'Smoking'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer: One possible reasoning behind the lack of causal relation between headache and smoking is that headache and smoking are both associated with various health conditions, but they do not directly cause or affect each other. Headache is a term that refers to pain or discomfort in the head, scalp, or neck, which can have many possible causes, such as stress, dehydration, infection, injury, or medication. Smoking is a habit that involves inhaling tobacco smoke, which contains nicotine and other harmful chemicals, which can increase the risk of diseases such as cancer, heart disease, stroke, and lung disease. Therefore, headache and smoking are not causally related, but they are both linked to different health problems. Therefore the answer is <Answer>C</Answer>

        Question: For a causal graph used for modeling factors causing Coronary Heart Diseases with the following nodes: ['Family Disease', 'Gene', 'Smoking', 'Blood Pressure', 'Coronary Heart Disease', 'Headache'],
            Which cause-and-effect relationship is more likely between nodes 'Coronary Heart Disease' and 'Smoking'?

            A. changing the state of node 'Smoking' causally effects a change in another node 'Coronary Heart Disease'.

            B. changing the state of node 'Coronary Heart Disease' causally effects a change in another node 'Smoking'.

            C. There is no causal relation between the nodes 'Coronary Heart Disease' and 'Smoking'.

        Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
        It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer: Possible reasoning behind the causal effect of smoking on coronary heart disease is smoking damages the heart and blood vessels by raising triglycerides, lowering HDL, increasing blood clotting, and impairing blood flow to the heart. This can lead to plaque buildup, heart attacks, and death. Therefore answer is
        <Answer>A</Answer>. 
        
        
        Question: For a causal graph used {context} with the following nodes: {nodes}, 
        Which cause-and-effect relationship is more likely between nodes {X} and {Y}?

        A. changing the state of node {X} causally effects a change in another node {Y}.

        B. changing the state of node {Y} causally effects a change in another node {X}.

        C. There is no causal relation between the nodes {X} and {Y}.

        Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
        It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.
        '''

        messages = [{"role": "system", "content": "You are an expert in causal reasoning. You are able to accurately judge direction of edges in causal graphs."},
        {"role": "assistant", "content": f"{input_prompt}"}]

        # llm_client = LLMClient()
        request_data = {
                "messages":messages,
                "max_tokens":700,
                "temperature":0,
                "top_p":1,
                "frequency_penalty":0,
                "presence_penalty":0
        }
    
        try:
        #     if model_name in CHATMODELS:
        #         # messages = []
        #         # if system:
        #         #     messages.append({"role": "system", "content": system})
        #         # messages.append({"role": "user", "content": prompts[i]})
        #         if openai.api_type == "azure":
        #             response = openai.ChatCompletion.create(
        #                 engine=model_name, 
        #                 messages=messages,
        #                 temperature=0)
        #         else:
        #             response = openai.ChatCompletion.create(
        #                 model=model_name, 
        #                 messages=messages,
        #                 temperature=0)
        #     else:
        #         response = openai.Completion.create(
        #                 model=model_name,
        #                 prompt=prompts[i], # type: ignore
        #                 temperature=1e-3,
        #                 max_tokens=512)
        #                 #top_p=1,
        #                 #frequency_penalty=0,
        #                 #presence_penalty=0,
        #             #logprobs=5)
        #     # prompts[i]["result"] = response
        #     print(response)
        #     print('\n')
        #     print('######################################')
        #     answer = response['choices'][0]['message']['content'].replace(' .', '.').strip() # type: ignore
        #     print('\n')
        #     print(answer)

        #     result = re.search('<Answer>(.*?)</Answer>', answer)
        #     print(result)
            print ("running")
            
            if result:
                print(result.group(1))
                ans = result.group(1)
                if ans == 'A':
                    return "A_2_B"
                elif ans == 'B':
                    return "B_2_A"
                else:
                    return "No_Conn"
            else:
                print("Format not followed!!")
            # time.sleep(100)
            
        except Exception as e:
            print(e)
            sleep_time = 5
            print(f"Exceeded Rate Limit. Waiting for {sleep_time} seconds")
            time.sleep(sleep_time)
            

            
    def subgraph_merge(self, subgroup_list, subgraph_list, nodes, context):

        final_graph = []
        #Calculating Prob. Distribution
        edgewise_dist = {}
        for i in tqdm(range(0, len(nodes))):
            x = nodes[i]
            for j in tqdm(range(i+1, len(nodes))):
                y = nodes[j]
                x_y_list = []
                y_x_list = []
                x_y_ind = []
                print('X: ', x)
                print('\n')
                print('Y: ', y)
                print('\n')
                #find subgraphs containing x and y
                
                subgraph_index = []
                xy_group = [triplet for triplet in subgroup_list if x in triplet and y in triplet]
                for ele in xy_group:
                    subgraph_index.append(subgroup_list.index(ele))
                    
                # pdb.set_trace()
                for index in tqdm(subgraph_index):
                    print("INDEX: ", index)
                    print('\n')
                    x_y_list.append([item for item in subgraph_list[index] if item == (x, y)])
                    x_y_list = [x for x in x_y_list if x != []]
                    y_x_list.append([item for item in subgraph_list[index] if item == (y, x)])
                    y_x_list = [x for x in y_x_list if x != []]
                    # x_y_ind.append([item for item in subgraph_list[index] if item == (x) and item == (y)])
                    # x_y_ind.append([item for item in subgraph_list[index] if (x,y) not in subgraph_list[index] and 
                    # (y,x) not in subgraph_list[index]])

                    sublist = [item for item in subgraph_list[index] if (x, y) not in subgraph_list[index] and (y, x) not in subgraph_list[index]]

                    if sublist:
                        x_y_ind.append(sublist)

                lists = {
                        "A_2_B": x_y_list,
                        "B_2_A": y_x_list,
                        "No_Conn": x_y_ind
                    }
                
                total_den = len(x_y_list) + len(y_x_list) + len(x_y_ind)
                edgewise_dist[(nodes[i],nodes[j])] = [len(x_y_list)/total_den, len(y_x_list)/total_den, len(x_y_ind)/total_den]
                
                for key, value in lists.items():
                    print("Lenght of list of each Key in dictionary: ",)
                    print(key, len(value))
                    print('\n')
                    print(value)
                
                max_len = max(len(v) for v in lists.values())
                keys = [k for k,v in lists.items() if len(v) == max_len]

                if len(keys) == 1:
                    max_key = keys[0]
                else:
                    # tie_breaker = WeakLearners.query_gpt4(self,'gpt-4',x + '(' + sangiovese_var_dict.get(x) + ')',y + '(' + sangiovese_var_dict.get(y) + ')',nodes, context)
                    tie_breaker = WeakLearners.query_gpt4(self,'gpt-4',x,y,nodes, context)
                    max_key = tie_breaker

                
                # max_key = max(lists, key=lambda x: len(lists[x]))

                if max_key == "A_2_B":
                    final_graph.append((x, y))
                    print("Max is A_2_B")
                elif max_key == "B_2_A":
                    final_graph.append((y, x))
                    print("Max is B_2_A")
                else:
                    print("No connection between ", x, " and ", y)
                    print('\n')

        return final_graph, edgewise_dist
    
    
    def str_2_lst(self, str1):
        lst1 = ast.literal_eval(str1)
        lst1 = [ele if isinstance(ele, tuple) else (ele,) for ele in lst1]
        return lst1
    
    
    def str_filter_lst(self, str1):
        pattern = r"\[.*?\]|\(.*?\)"

        result = re.findall(pattern, str1)
        result = [eval(x) for x in result]

        return(result[0])

            


if __name__ == "__main__":
    
   
####################################################################################################
    covid_nodes = ['Virus Enters Upper Respiratory Tract', 'Upper Respiratory Tract epithelial infection', 'Infection of olfactory epithelium','Anosmia and/or aguesia', 'Alveolar epithelial infection', 'Alveolar endothelial infection','Viremia', 'Systemic immune/inflammatory response','Pulmonary capillary leakage', 'Dry cough', 'Productive cough']

    covid_graph = [('Virus Enters Upper Respiratory Tract', 'Upper Respiratory Tract epithelial infection'), 
                         ('Virus Enters Upper Respiratory Tract','Alveolar epithelial infection'),
                         ('Upper Respiratory Tract epithelial infection', 'Alveolar epithelial infection'),
                         ('Upper Respiratory Tract epithelial infection', 'Infection of olfactory epithelium'),
                         ('Upper Respiratory Tract epithelial infection', 'Dry cough'),
                         ('Upper Respiratory Tract epithelial infection', 'Systemic immune/inflammatory response'),
                         ('Upper Respiratory Tract epithelial infection', 'Viremia'),
                         ('Infection of olfactory epithelium', 'Anosmia and/or aguesia'), 
                         ('Alveolar epithelial infection', 'Productive cough'),
                         ('Alveolar epithelial infection', 'Pulmonary capillary leakage'),
                         ('Alveolar epithelial infection', 'Systemic immune/inflammatory response'),
                         ('Alveolar epithelial infection', 'Viremia'),
                         ('Alveolar epithelial infection', 'Alveolar endothelial infection'),
                         ('Alveolar endothelial infection', 'Pulmonary capillary leakage'),
                         ('Alveolar endothelial infection', 'Systemic immune/inflammatory response'),
                         ('Alveolar endothelial infection', 'Viremia'),
                         ('Pulmonary capillary leakage', 'Productive cough'),
                         ('Pulmonary capillary leakage', 'Dry cough'),
                         ('Systemic immune/inflammatory response', 'Pulmonary capillary leakage'),
                         ('Viremia', 'Systemic immune/inflammatory response')]

###########################################################################################
    gt_graph_asia = [('visit to Asia', 'tuberculosis'), ('tuberculosis', 'either tuberculosis or lung cancer'),('either tuberculosis or lung cancer', 'positive X-ray'),('smoking', 'lung cancer'),('smoking', 'bronchitis'), ('lung cancer', 'either tuberculosis or lung cancer'), ('either tuberculosis or lung cancer', 'dyspnoea'), ('bronchitis', 'dyspnoea')]
    
    Asia_nodes = ['visit to Asia', 'tuberculosis', 'either tuberculosis or lung cancer', 
                  'positive X-ray', 'dyspnoea', 'bronchitis', 'lung cancer', 'smoking']

    Asia_var_dict = {'visit to Asia':'visiting Asian countries with high exposure to pollutants',
                    'tuberculosis': 'tuberculosis',
                    'either tuberculosis or lung cancer': 'tuberculosis or lung cancer', 
                    'positive X-ray':'getting positve xray result',
                    'dyspnoea':'dyspnoea',
                    'bronchitis': 'bronchitis',
                    'lung cancer': 'lung cancer',
                    'smoking': 'smoking habit'}
###########################################################################################
    alz_nodes = ['sex', 'age', 'ventricular volume', 'brain volume', 'av45', 'tau', 'brain MRI', 'slice number', 'education', 'moca', 'APOE4']

    alz_graph = [('sex', 'ventricular volume'),
                 ('sex', 'brain volume'),
                 ('age', 'ventricular volume'),
                 ('age', 'brain volume'),
                 ('age', 'av45'),
                 ('age', 'tau'),
                 ('age', 'moca'),
                 ('ventricular volume', 'brain MRI'),
                 ('brain volume', 'ventricular volume'),
                 ('brain volume', 'moca'),
                 ('brain volume', 'ventricular volume'),
                 ('brain volume', 'brain MRI'),
                 ('brain volume', 'moca'),
                 ('av45', 'brain volume'),
                 ('av45', 'tau'),
                 ('tau', 'moca'),
                 ('tau', 'ventricular volume'),
                 ('tau', 'brain volume'),
                 ('slice number', 'brain MRI'),
                 ('education', 'moca'),
                 ('APOE4', 'av45')]
    
    alz_descr = {'APOE4': 'Expression level of APOE4 gene','sex':'Biological Sex of Patient', 'age': 'Age of Patient', 'education': 'Educational attainment (years)', 'av45':'Beta Amyloid protein level measured by Florbetapir F18', 'tau': 'Phosphorylated-tau deposition', 'brain volume':'Total Brain Matter Volume of Patient', 'ventricular volume': 'Total Ventricular Volume of Patient', 'moca': ' Montreal Cognitive Assessment Score'}
##################################################################################
    Child_nodes = ['BirthAsphyxia', 'HypDistrib', 'HypoxiaInO2', 'CO2', 'ChestXray', 'Grunting', 'LVHreport', 'LowerBodyO2', 
                   'RUQO2', 'CO2Report', 'XrayReport', 'Disease', 'GruntingReport', 'Age', 'LVH', 'DuctFlow', 'CardiacMixing', 
                   'LungParench', 'LungFlow', 'Sick']
    
    
    ground_truth_graph_child = [('BirthAsphyxia', 'Disease'), ('HypDistrib', 'LowerBodyO2'), ('HypoxiaInO2', 'LowerBodyO2'), 
                                ('HypoxiaInO2', 'RUQO2'), ('CO2', 'CO2Report'), ('ChestXray', 'XrayReport'), ('Grunting', 'GruntingReport'),
                                  ('Disease', 'Age'), ('Disease', 'LVH'), ('Disease', 'DuctFlow'), ('Disease', 'CardiacMixing'), 
                                  ('Disease', 'LungParench'), ('Disease', 'LungFlow'), ('Disease', 'Sick'), ('LVH', 'LVHreport'), 
                                  ('DuctFlow', 'HypDistrib'), ('CardiacMixing', 'HypDistrib'), ('CardiacMixing', 'HypoxiaInO2'), 
                                  ('LungParench', 'HypoxiaInO2'), ('LungParench', 'CO2'), ('LungParench', 'ChestXray'), 
                                  ('LungParench', 'Grunting'), ('LungFlow', 'ChestXray'), ('Sick', 'Grunting'), ('Sick', 'Age')]
    
    child_var_dict = {'BirthAsphyxia':"lack of oxygen to the blood during the infant's birth", 
                      'HypDistrib':"low oxygen areas equally distributed around the body",
                    'HypoxiaInO2':"hypoxia when breathing oxygen",
                    'CO2':"level of carbon dioxide in the body",
                    'ChestXray':"having a chest x-ray",
                    'Grunting':"grunting in infants",
                    'LVHreport':"report of having left ventricular hypertrophy",
                    'LowerBodyO2':"level of oxygen in the lower body",
                    'RUQO2':"level of oxygen in the right up quadricep muscule",
                    'CO2Report':"a document reporting high level of CO2 levels in blood",
                    'XrayReport':"lung excessively filled with blood",
                    'Disease':"infant methemoglobinemia",
                    'GruntingReport':"report of infant grunting",
                    'Age':"age of infant at disease presentation",
                    'LVH':"thickening of the left ventricle",
                    'DuctFlow':"blood flow across the ductus arteriosus",
                    'CardiacMixing':"mixing of oxygenated and deoxygenated blood",
                    'LungParench':"the state of the blood vessels in the lungs",
                    'LungFlow':"low blood flow in the lungs",
                    'Sick':"presence of an illness"}



##################################################################################
    backoff_ceil = 10
    backoff_base = 2
    backoff_count = 0
    
    sys.stdout = open('/home/t-aniketva/Desktop/Causal_LLM_Neurips/triplet_results_phi/Child_Triplet_Voting_Merging_Phi.txt', 'w')
    
    WL = WeakLearners()
    global_context = 'model congenital heart disease in babies'
    # "modeling the clinical and radiological phenotype of Alzheimer's Disease"
    # 'model congenital heart disease in babies'
    # 'model the possible respiratory problems someone can have who has recently visited Asia and is experiencing shortness of breath'
    # 'modeling the initial pathophysiological process of SARS-CoV-2 in the respiratory system involves outlining the various pathways from viral infection to key complications.'

    nodes = Child_nodes
    gt_graph = ground_truth_graph_child
    descr_nodes = child_var_dict

    subgroup_list = WL.subgroup_from_nodes(nodes)
    print(subgroup_list)

    subgr_dict = dict()
    result_subgraphs_list = []
    CG = CausalGraph(nodes)

    for group in tqdm(subgroup_list):
        print('\n')
        print('Nodes being inspected: ', group)
        print('\n')
        descr_nodes_subdict = {key: descr_nodes[key] for key in group if key in descr_nodes}
        print("Description subdict:", descr_nodes_subdict)
        rerun = True
        while rerun:
            try:
                time.sleep(12)
               
                messages = CG.generate_graphs_from_nodes_w_descr(nodes = group, context = global_context, descr_nodes=descr_nodes_subdict)
                
                # response = client.chat.completions.create(model='gpt-35-turbo',
                #                                         temperature=0.0,
                #                                         max_tokens=400,
                #                                         messages=messages)
                
                output = pipe(messages, **generation_args)
                print("Answer:", output)                
                print('\n')
                gen = output[0]['generated_text']
                print(gen)
                result = re.search('<Answer>(.*?)</Answer>', gen)

                # answer = response['choices'][0]['message']['content'].replace(' .', '.').strip() # type: ignore
                # answer = response.choices[0].message.content
                # print("Answer:", answer)

                # result = re.search('<Answer>(.*?)</Answer>', answer)
                
                ans = result.group(1)
                # final_dag = ast.literal_eval(result)
                print("Final DAG: ", ans)
                result_subgraphs_list.append(ans)
                print('\n')
                print('#################################')
                print('Final list of oriented subgraphs: ', result_subgraphs_list)
                print('#################################')
                print('\n')
                subgr_dict[group] = ans
                break
            except Exception as e:
                print(e)
                backoff_count = min(backoff_count + 1, backoff_ceil)
                sleep_time = backoff_base**backoff_count
                print(f"Exceeded Rate Limit. Waiting for {sleep_time} seconds")
                time.sleep(sleep_time)
                continue
        
        
    # # ###############################################################################################################
    #Final dict with triplet groups and corresponding dags
    

    print("Subgraph list: ",len(result_subgraphs_list))
    subgroup_list = list(subgr_dict.keys())
    print('\n')
    print('Final list of oriented subgraphs: ', result_subgraphs_list)
    print('\n')
    final_list = [WL.str_2_lst(x) for x in result_subgraphs_list]
    # Only while doing manual intervention#
    # nodes = list(child_var_dict.values())
    # subgroup_list = WL.subgroup_from_nodes(nodes)
    final_merged_graph, edgewise_dist = WL.subgraph_merge(subgroup_list, final_list, nodes, context = global_context) 
    CG = CausalGraph(nodes = nodes)
    conn_nodes_LLM_graph = list(set([i for j in final_merged_graph for i in j]))
    print("Final Merged Graph is ", final_merged_graph)
    print('\n')
    print("Connected nodes in the final merged graph are: ", conn_nodes_LLM_graph)
    print('No. of connected nodes in the final merged graph are: ', len(conn_nodes_LLM_graph))
    CG = CausalGraph(nodes)
    # # llm_order = CG.topological_ordering(list(final_merged_graph))
    # # TD_G = CG.d_top_score(llm_order,list(gt_graph)) 

    #Dumping edgewise probability distribution in pickle
    with open('/home/t-aniketva/Desktop/Causal_LLM_Neurips/triplet_results_phi/Child_edgewise_score_phi.pkl', 'wb') as file:
        # Dump the dictionary into the file
        pickle.dump(edgewise_dist, file)


    SHD = CG.structural_hamming_distance(list(gt_graph), list(final_merged_graph), nodes)
    print("SHD of Final merged graph:", SHD)


    unique_elements = list(set(item for sublist in final_merged_graph for item in sublist))
    print('Connected nodes:',unique_elements)
    print('Number of isolated nodes:',len(nodes)-len(unique_elements))
    
    G = nx.DiGraph(final_merged_graph)
    # print("Number of k lenght cycles:",len(sorted(nx.simple_cycles(G, length_bound = 10))))
    list_of_lists = sorted(nx.simple_cycles(G))

    dict_result = {}
    for sublist in list_of_lists:
        tuple_key = tuple(sublist)
        dict_result[tuple_key] = "some_value"  # You can replace "some_value" with the desired value for each key

    # print(dict_result)
    print("Number of cycles:",len(dict_result))

    print('\n')
    print("Edgewise score distribution:",edgewise_dist)
    print('\n')


    TD_2 = CG.topological_divergence_2(gt_graph, list(final_merged_graph))
    print("TD of Final merged graph:", TD_2)