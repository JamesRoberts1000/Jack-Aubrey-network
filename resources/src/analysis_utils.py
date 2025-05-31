# src/analysis_utils.py
"""
Utility functions for character and relationship analysis in the Aubrey-Maturin NLP project.
"""
import pandas as pd
import matplotlib.pyplot as plt
import spacy

# --- Book and NLP Utilities ---
def load_and_process_book(file_path, nlp_model):
    """
    Read a text file and process it with a spaCy model.
    Args:
        file_path (str): Path to the text file.
        nlp_model (spacy.lang): Loaded spaCy language model.
    Returns:
        spacy.tokens.Doc: The processed spaCy document.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return nlp_model(text)

def visualize_entities(doc, n=200):
    """
    Visualize named entities in the first n tokens of a spaCy document using displacy.
    Args:
        doc (spacy.tokens.Doc): The spaCy-processed document.
        n (int): Number of tokens from the start of the document to visualize.
    Returns:
        None. Displays the visualization in the notebook.
    """
    from spacy import displacy
    displacy.render(doc[:n], style="ent", jupyter=True)

# --- DataFrame and Entity Utilities ---
def filter_sentences_with_entities(sent_entity_df):
    """
    Return only sentences that contain at least one entity.
    Args:
        sent_entity_df (pd.DataFrame): DataFrame with an 'entities' column.
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    return sent_entity_df[sent_entity_df['entities'].map(len) > 0]

def extract_first_names(entity_list):
    """
    Extract first names from a list of full names.
    Args:
        entity_list (list): List of full name strings.
    Returns:
        list: List of first names.
    """
    return [item.split()[0] for item in entity_list]

# --- NetworkX Graph Utilities ---
def plot_networkx_graph(G, figsize=(10,10)):
    """
    Plot a NetworkX graph using matplotlib.
    Args:
        G (networkx.Graph): The graph to plot.
        figsize (tuple): Figure size.
    Returns:
        None. Displays the plot.
    """
    import networkx as nx
    plt.figure(figsize=figsize)
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos)
    plt.show()

# --- Character and Relationship Extraction ---
def normalize_name(name):
    """
    Normalize a character name by converting it to lowercase, removing possessive 's, stripping punctuation, and trimming whitespace.
    Args:
        name (str): The character name to normalize.
    Returns:
        str: The normalized character name.
    """
    import re
    name = name.lower()
    name = re.sub(r"'s\\b", "", name)  # Remove possessive 's
    name = re.sub(r"[^\w\s]", "", name)  # Remove punctuation
    name = name.strip()
    return name

def filter_entity(ent_list, character_df):
    """
    Function to filter out non-character entities.
    Args:
        ent_list -- list of entities to be filtered
        character_df -- a dataframe contain characters' names and characters' first names
    Returns:
        a list of entities that are characters (matching by names or first names).
    """
    return [ent for ent in ent_list if ent in list(character_df.character)]

def get_ent_list_per_sentence(spacy_doc):
    """
    Get a list of entities per sentence of a Spacy document and store in a dataframe.
    Args:
        spacy_doc -- a Spacy processed document
    Returns:
        a dataframe containing the sentences and corresponding list of recognised named entities in the sentences
    """
    sent_entity_df = []
    for sent in spacy_doc.sents:
        entity_list = [ent.text for ent in sent.ents]
        sent_entity_df.append({"sentence": sent, "entities": entity_list})
    sent_entity_df = pd.DataFrame(sent_entity_df)
    return sent_entity_df

def create_relationships(df, window_size):
    """
    Create a dataframe of relationships based on the df dataframe (containing lists of characters per sentence) and the window size of n sentences.
    Args:
        df -- a dataframe containing a column called character_entities with the list of characters for each sentence of a document.
        window_size -- size of the windows (number of sentences) for creating relationships between two adjacent characters in the text.
    Returns:
        a relationship dataframe containing 3 columns: source, target, value.
    """
    import numpy as np
    relationships = []
    for i in range(df.index[-1]):
        end_i = min(i + window_size, df.index[-1])
        char_list = sum((df.loc[i: end_i].character_entities), [])
        char_unique = [char_list[i] for i in range(len(char_list)) if (i==0) or char_list[i] != char_list[i-1]]
        if len(char_unique) > 1:
            for idx, a in enumerate(char_unique[:-1]):
                b = char_unique[idx + 1]
                relationships.append({"source": a, "target": b})
    relationship_df = pd.DataFrame(relationships)
    relationship_df = pd.DataFrame(np.sort(relationship_df.values, axis = 1), columns = relationship_df.columns)
    relationship_df["value"] = 1
    relationship_df = relationship_df.groupby(["source","target"], sort=False, as_index=False).sum()
    return relationship_df

def ner(file_name):
    """
    Function to process text from a text file (.txt) using Spacy.
    Args:
        file_name -- name of a txt file as string
    Returns:
        a processed doc file using Spacy English language model
    """
    nlp = spacy.load("en_core_web_sm")
    book_text = open(file_name).read()
    book_doc = nlp(book_text)
    return book_doc

def get_named_entities_per_sentence(doc):
    """
    Extracts a list of PERSON named entities for each sentence in a spaCy document.
    Args:
        doc (spacy.tokens.Doc): The spaCy-processed document.
    Returns:
        list: A list where each element is a list of PERSON entity strings found in a sentence.
    """
    sentences = list(doc.sents)
    named_entities = []
    for sentence in sentences:
        ents = [ent.text for ent in sentence.ents if ent.label_ == 'PERSON']
        named_entities.append(ents)
    return named_entities
