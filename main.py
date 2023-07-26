from typing import Optional# from transformers import AutoTokenizer, AutoModelForTokenClassification# from transformers import pipeline# from gutenberg.acquire import load_etext# from gutenberg.cleanup import strip_headersimport spacyfrom fastcoref import LingMessCorefimport matplotlib.pyplot as pltfrom pyvis.network import Networkfrom narrative import CinderellaNarrativeimport requestsimport jsonfrom transformers import RobertaTokenizerFast, RobertaModelimport torchfrom distinctipy import distinctipydef get_cinderella(book_id: int):    text = strip_headers(        load_etext(book_id,                   mirror="http://www.mirrorservice.org/sites/ftp.ibiblio.org/pub/docs/books/gutenberg/")    ).strip()    return textclass Cinderella(object):    def __init__(self, uid: int):        self.uid = uiddef get_index_of_contradiction(lst):    indices = []    for i, value in enumerate(lst):        if value == "contradiction":            indices.append(i)    return indicesdef event_segmentation(sentences):    prompt = "An event is an ongoing coherent situation. " \             "The following story needs to be copied and segmented into events. " \             "Copy the following story word-for-word and start a new line " \             "whenever one event ends and another begins. This is the story: "    message = prompt + sentences    headers = {'Content-Type': 'application/json'}    payload = {"model": "vicuna-13b-v1.1",               "messages": [{"role": "user", "content": message}],               "temperature": 0,               "key": "M7ZQL9ELMSDXXE86"}    response = requests.post('https://turbo.skynet.coypu.org', headers=headers, json=payload)    return json.loads(response.text)def instantiate_ner(model: Optional = "spacy_sm"):    if model == "spanmaker":        nlp = spacy.load("en_core_web_trf", exclude=["ner"])        nlp.add_pipe("span_marker", config={"model": "tomaarsen/span-marker-roberta-large-ontonotes5"})    elif model == "spacy_trf":        nlp = spacy.load("en_core_web_trf")    elif model == "spacy_sm":        nlp = spacy.load("en_core_web_sm")    else:        raise NotImplementedError    return nlpdef run_ner_spacy(model, sentences):    doc = model(sentences)    ner_results = []    for ent in doc.ents:        ner_results.append([ent.text, ent.start_char, ent.end_char, ent.label_])    ner_person = list(set([ner[0] for ner in ner_results if ner[-1] == "PERSON"]))    return ner_results, ner_persondef instantiate_crr(model: Optional = "spacy"):    if model == "spacy":        crr = spacy.load("en_core_web_sm")        crr.add_pipe("fastcoref", config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cpu'})        return crr    if model == "fastcoref":        crr = LingMessCoref(device='cpu')        return crrdef run_crr(model, sentences):    preds = model.predict(        texts=[sentences]    )    for pred in preds:        print(pred.get_clusters(as_strings=False))        print(pred.get_clusters())    print(preds)    return predsdef rgb_to_hex(rgb):    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))def create_colored_html(text, resolved_mentions):    # Create a list of unique coreference clusters    clusters = set()    for _, cluster_id in resolved_mentions:        clusters.add(cluster_id)    # Generate N distinct colors using distinctipy    num_colors = len(clusters)    rgb_colors = distinctipy.get_colors(num_colors)    # Convert RGB colors to hexadecimal format    colors = [rgb_to_hex(rgb) for rgb in rgb_colors]    # Create a dictionary to map cluster IDs to colors    cluster_color_map = dict(zip(clusters, colors))    resolved_mentions = sorted(resolved_mentions, key=lambda x: x[0][0])    # Create HTML content with colored mentions    html_content = "<div style='font-size: 16px; line-height: 1.5;'>"    current_index = 0    for mention, cluster_id in resolved_mentions:        start, end = mention        mention_text = text[start:end]        # Add non-coreferent text (if any) before the mention        non_coreferent_text = text[current_index:start]        html_content += non_coreferent_text        # Add the mention with the corresponding color        color = cluster_color_map[cluster_id]        html_content += f"<span style='color: {color};'>{mention_text}</span>"        current_index = end    # Add any remaining non-coreferent text    remaining_text = text[current_index:]    html_content += remaining_text    html_content += "</div>"    return html_contentdef convert_character_spans_to_sentence_indices(document, coref_resolution_results):    # Load the spaCy English model for sentence splitting    nlp = spacy.load("en_core_web_sm")    # Parse the document into sentences    doc = nlp(document)    sentences = list(doc.sents)    # Create a list to store the sentence indices for each cluster    cluster_sentence_indices = []    for cluster_label, character_spans in coref_resolution_results.items():        sentence_indices = []        for start_char, end_char in character_spans:            # Find the sentence index for each character span            for sent_index, sentence in enumerate(sentences):                if sentence.start_char <= start_char < sentence.end_char and \                   sentence.start_char <= end_char < sentence.end_char:                    sentence_indices.append(sent_index)        cluster_sentence_indices.append(sentence_indices)    return cluster_sentence_indicesdef get_mean_aggregated_embedding(nlp, roberta_tokenizer, roberta_model, document, coref_resolution_results):    cluster_sentence_indices = convert_character_spans_to_sentence_indices(document, coref_resolution_results)    # Tokenize all sentences together to obtain token-level information    sentence_list = [str(sent) for sent in nlp(document).sents]    tokenized_sentences = roberta_tokenizer(        sentence_list,        return_tensors="pt",        padding=True,        truncation=True,    )    # Get the hidden states for all sentences together    with torch.no_grad():        outputs = roberta_model(**tokenized_sentences)    # Get the mention embeddings from the hidden states    mention_embeddings = []    for sentence_indices, character_spans in zip(cluster_sentence_indices, coref_resolution_results.values()):        for character_span in character_spans:            start_char, end_char = character_span            # Find the sentence index for the character span            sent_index = None            for index, sentence in enumerate(nlp(document).sents):                if sentence.start_char <= start_char < sentence.end_char and \                        sentence.start_char <= end_char < sentence.end_char:                    sent_index = index                    break            if sent_index is not None:                # Get the token indices for the sentence containing the mention                mention_indices = list(range(outputs.last_hidden_state.shape[1]))  # All token indices                mention_embedding = outputs.last_hidden_state[sentence_indices[0], mention_indices, :].mean(dim=0)                mention_embeddings.append(mention_embedding)    return mention_embeddingsdef preprocess_story():    from fastcoref import spacy_component    roberta_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")    roberta_model = RobertaModel.from_pretrained("roberta-base")    ## instantiation    cinderella = CinderellaNarrative(path="narratives/cinderella/23303.txt")    # ner = instantiate_ner("spacy_sm")    crr = instantiate_crr("fastcoref")    nlp = spacy.load("en_core_web_trf")    ## NER    doc = nlp(cinderella.preprocessed_text)    ner_results = []    for ent in doc.ents:        ner_results.append([ent.text, ent.start_char, ent.end_char, ent.label_])    ner_person = list(set([ner[0] for ner in ner_results if ner[-1] == "PERSON"]))    # ner_results, ner_person = run_ner_spacy(ner, cinderella.preprocessed_text)    print(ner_results)    print(ner_person)    # print("coref clusters", doc._.coref_clusters)    ## CRR    predictions = run_crr(crr, cinderella.preprocessed_text)    clusters = [prediction.get_clusters() for prediction in predictions][0]    spans = [prediction.get_clusters(as_strings=False) for prediction in predictions][0]    crr_dict = {person: [] for person in ner_person}    resolved_mentions = []    for i, cluster in enumerate(clusters):        for j, mention in enumerate(cluster):            resolved_mentions.append((list(spans[i][j]), i))        for person in ner_person:            if person in cluster:                crr_dict[person].extend(spans[i])    print(crr_dict)    print(resolved_mentions)    html_content = create_colored_html(cinderella.preprocessed_text, resolved_mentions)    # Save the HTML content to a file or display it in a web browser    with open("output.html", "w") as f:        f.write(html_content)    cluster_sentence_indices = convert_character_spans_to_sentence_indices(cinderella.preprocessed_text, crr_dict)    # Print the list of sentence indices for each cluster    for i, sentence_indices in enumerate(cluster_sentence_indices):        print(f"Cluster {i}: Sentence Indices = {sentence_indices}")    mention_embeddings = get_mean_aggregated_embedding(nlp, roberta_tokenizer, roberta_model, cinderella.preprocessed_text, crr_dict)    for i, embedding in enumerate(mention_embeddings):        print(f"Mention {i} - Embedding shape: {embedding.shape}")def toy_temporality_prototype():    cinderella_1 = CinderellaNarrative(path="narratives/cinderella/20723.txt")    cinderella_2 = CinderellaNarrative(path="narratives/cinderella/10830.txt")    cinderella_3 = CinderellaNarrative(path="narratives/cinderella/23303.txt")    print(f"Cinderella 1 sentence count: {cinderella_1.sentence_count}")    print(f"Cinderella 2 sentence count: {cinderella_2.sentence_count}")    print(f"Cinderella 3 sentence count: {cinderella_3.sentence_count}")    event_segments = event_segmentation(cinderella_1.preprocessed_text)    print(event_segments)    g = Network(height="1100px", width="100%", bgcolor="#FAEBD7", font_color="#3D2B1F", notebook=True)    g.add_nodes(list(range(cinderella_1.sentence_count)),                size=[50]*cinderella_1.sentence_count,                title=cinderella_1.segmented_text,                label=[str(index) for index in list(range(cinderella_1.sentence_count))],                color=["#F4C2C2"]*cinderella_1.sentence_count                )    start = cinderella_1.sentence_count    g.add_nodes(list(range(start, start+cinderella_2.sentence_count)),                size=[50]*cinderella_2.sentence_count,                title=cinderella_2.segmented_text,                label=[str(index) for index in list(range(cinderella_2.sentence_count))],                color=["#89CFF0"] * cinderella_2.sentence_count                )    start = cinderella_1.sentence_count+cinderella_2.sentence_count    g.add_nodes(list(range(start, start+cinderella_3.sentence_count)),                size=[50]*cinderella_3.sentence_count,                title=cinderella_3.segmented_text,                label=[str(index) for index in list(range(cinderella_3.sentence_count))],                color=["#E9D66B"] * cinderella_3.sentence_count                )    embedder = SentenceTransformer('all-MiniLM-L6-v2')    cinderella_1_embeddings = embedder.encode(cinderella_1.segmented_text, convert_to_tensor=True)    cinderella_2_embeddings = embedder.encode(cinderella_2.segmented_text, convert_to_tensor=True)    cinderella_3_embeddings = embedder.encode(cinderella_3.segmented_text, convert_to_tensor=True)    cinderella_1_index = []    cinderella_2_index = []    cinderella_3_index = []    similarity_1 = []    similarity_2 = []    for i, query in enumerate(cinderella_1_embeddings):        top_hit = util.semantic_search(query, cinderella_2_embeddings, top_k=1)[0][0]        top_hit_2 = util.semantic_search(query, cinderella_3_embeddings, top_k=1)[0][0]        cinderella_1_index.append(i)        cinderella_2_index.append(top_hit['corpus_id'])        cinderella_3_index.append(top_hit_2['corpus_id'])        similarity_1.append(top_hit['score'])        similarity_2.append(top_hit_2['score'])    # plot the frequency of similarities    # plt.hist(similarity_1, bins=np.arange(min(similarity_1), max(similarity_1)+1)-0.5)    # plt.savefig("similarity_1.png")    # plt.hist(similarity_2, bins=np.arange(min(similarity_2), max(similarity_2)+1)-0.5)    # plt.savefig("similarity_2.png")    # counts = Counter(similarity_1)    # plt.bar(x=counts.keys(), height=[a for a in counts.values()], width=3e-4)    # plt.savefig("similarity_1.png")    # counts = Counter(similarity_2)    # plt.bar(x=counts.keys(), height=[a / sum(counts.values()) for a in counts.values()], width=3e-4)    # plt.savefig("similarity_2.png")    fig = plt.figure()    ax = fig.add_subplot(111)    n, bins, rectangles = ax.hist(similarity_1, 15, density=True)    fig.canvas.draw()    fig.savefig("similarity_1.png")    fig = plt.figure()    ax = fig.add_subplot(111)    n, bins, rectangles = ax.hist(similarity_2, 15, density=True)    fig.canvas.draw()    fig.savefig("similarity_2.png")    # NLI    model = CrossEncoder('cross-encoder/nli-deberta-v3-base')    cinderella_1_nli_data = [(cinderella_1.segmented_text[i], cinderella_1.segmented_text[i+1]) for i in range(len(cinderella_1.segmented_text) - 1)]    scores = model.predict(cinderella_1_nli_data)    labels = [['contradiction', 'entailment', 'neutral'][score_max] for score_max in scores.argmax(axis=1)]    contradiction_indices = get_index_of_contradiction(labels)    fig = plt.figure()    ax = fig.add_subplot(111)    ax.hist(contradiction_indices, 15, density=True)    fig.canvas.draw()    fig.savefig("contradiction.png")    # Get a list of nodes    node_list = g.get_nodes()    # Print the list of nodes    # print(node_list)    for i in contradiction_indices:        g.add_edge(i, i+1, color="red")    for i in range(len(cinderella_1_index)):        g.add_edge(cinderella_1_index[i], cinderella_1.sentence_count+cinderella_2_index[i],                   value=similarity_1[i], color="#ED872D", size=similarity_1[i],                   title=similarity_1[i], label=similarity_1[i])    for i in range(len(cinderella_1_index)):        g.add_edge(cinderella_1.sentence_count+cinderella_2_index[i],                   cinderella_1.sentence_count+cinderella_2_index[i]+cinderella_3_index[i],                   value=similarity_2[i], color="#ED872D", size=similarity_2[i],                   title=similarity_2[i], label=similarity_2[i])    g.barnes_hut(gravity=-200, central_gravity=0, spring_length=1000, spring_strength=0.001, damping=0.09, overlap=0)    for i in range(cinderella_1.sentence_count):        g.nodes[i]["x"] = i * 150        g.nodes[i]["y"] = 0    for i in range(cinderella_1.sentence_count, cinderella_1.sentence_count+cinderella_2.sentence_count):        g.nodes[i]["x"] = (i-cinderella_1.sentence_count) * 150        g.nodes[i]["y"] = -1500    start = cinderella_1.sentence_count+cinderella_2.sentence_count    for i in range(start, start+cinderella_3.sentence_count):        g.nodes[i]["x"] = (i-1.5*cinderella_1.sentence_count) * 150        g.nodes[i]["y"] = -3000    g.toggle_physics(False)    g.show("cinderella.html")if __name__ == "__main__":    # toy_temporality_prototype()    preprocess_story()    # tmp()    # from pyvis.network import Network    #    # # Create an empty network    # net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white",notebook=True)    #    # # Add nodes with empty attributes    # nodes = list(range(10))  # Example list of nodes from 0 to 9    # for node in nodes:    #     net.add_node(node)    #    # # Define the two lists indicating the connected nodes    # list1 = [0, 2, 4, 6, 8]  # Example list of indices    # list2 = [1, 3, 5, 7, 9]  # Example list of indices    #    # # Create edges between the connected nodes with weights    # for i in range(len(list1)):    #     net.add_edge(list1[i], list2[i], value=1, color="blue")    #    # # Set the layout to 'directed' for horizontal alignment    # net.barnes_hut(gravity=-200, central_gravity=0, spring_length=100, spring_strength=0.001, damping=0.09, overlap=0)    #    # # Manually position nodes in two separate lines    # for i, node in enumerate(list1):    #     net.nodes[node]["x"] = i * 100    #     net.nodes[node]["y"] = 0    #    # for i, node in enumerate(list2):    #     net.nodes[node]["x"] = i * 100    #     net.nodes[node]["y"] = -100    #    # # Generate the HTML file for visualization    # net.show("network.html")    # nlp = spacy.load("en_core_web_sm")    # nlp.add_pipe("fastcoref", config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cpu'})    #    # text = get_cinderella()    # print(text)    # clean_text = clean(text, fix_unicode=True, to_ascii=True, lower=False, no_line_breaks=True, no_urls=True, no_emails=True, lang="en")    #    # doc = nlp(  # for multiple texts use nlp.pipe    #     clean_text,    #     component_cfg={"fastcoref": {'resolve_text': True}}    # )    #    # resolved_text = doc._.resolved_text    # print(resolved_text)    # segmented_text = nltk.sent_tokenize(resolved_text)