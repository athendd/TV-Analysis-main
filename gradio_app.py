import gradio as gr
from theme_classifier import ThemeClassifier
from character_network import NamedEntityRecognizer, CharacterNetworkGenerator
from text_classification import JutsuClassifier
import os
from dotenv import load_dotenv
load_dotenv()

def get_themes(theme_list_str, subtitles_path, save_path):
    theme_list = theme_list_str.split(',')
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitles_path, save_path)
    
    theme_list = [theme for theme in theme_list if theme != 'dialogue']
    
    #Remove dialogue column from output_df
    output_df = output_df[theme_list]
    
    output_df = output_df[theme_list].sum().reset_index()
    output_df.columns = ['Theme', 'Score']
    
    output_chart = gr.BarPlot(
        output_df,
        x = 'Theme',
        y = 'Score',
        title = 'Series Themes',
        tooltip = ['Theme', 'Score'],
        vertical = False,
        width = 500,
        height = 260
    )
    
    return output_chart

def get_character_network(subtitiles_path, ner_path):
    ner = NamedEntityRecognizer()
    ner_df = ner.get_ners(subtitiles_path, ner_path)
    
    character_network_generator = CharacterNetworkGenerator()
    
    relationship_df = character_network_generator.generate_character_network(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)
    
    return html
    
def classify_text(text_classification_model, text_classification_data_path, text_to_classify):
    jutsu_classifier = JutsuClassifier(model_path = text_classification_model, data_path = text_classification_data_path,
                                       huggingface_token = os.getenv('huggingface_token'))
    
    output = jutsu_classifier.classify_justsu(text_to_classify)
    
    return output

def main():
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme Classification (Zero Shot Classifiers)</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot()
                    with gr.Column():
                        theme_list = gr.Textbox(label = "Themes")
                        subtitles_path = gr.Textbox(label = "Subtitles or Script Path")
                        save_path = gr.Textbox(label = "Save Path")
                        get_themes_button = gr.Button("Get Themes")
                        get_themes_button.click(get_themes, inputs = [theme_list, subtitles_path, save_path], outputs = [plot])
    
        #Character Network area   
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Text Classification with LLMs</h1>")
                with gr.Row():
                    with gr.Column():
                        text_classification_output = gr.Textbox(label = 'Text Classification Output')
                    with gr.Column():
                        text_classification_model = gr.Textbox(label = 'Model Path')
                        text_classification_data_path = gr.Textbox(label = 'Data Path')
                        text_to_classify = gr.Textbox(label = 'Text Input')
                        classify_text_button = gr.Button('Classify Text (Jutsu)')
                        classify_text_button.click(classify_text, inputs = [text_classification_model, text_classification_data_path, text_to_classify], outputs = [text_classification_output])
                        
        #Text classification with LLMs  
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Network(NERs and Graphs)</h1>")
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                    with gr.Column():
                        subtitles_path = gr.Textbox(label = "Subtitles or Script Path")
                        ner_path = gr.Textbox(label = "NERs Save Path")
                        get_network_graph_button = gr.Button("Get Character Network")
                        get_network_graph_button.click(get_character_network, inputs = [subtitles_path, ner_path], outputs = [network_html])
                            
    iface.launch(share = True)
                
    
if __name__ == '__main__':
    main()
