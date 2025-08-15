import gradio as gr
from theme_classifier import ThemeClassifier
from episode_summarizer import EpisodeSummarizer
from character_network import NamedEntityRecognizer, CharacterNetworkGenerator
from text_classification import NenClassifier
from character_chatbot import CharacterChatBot
import os
from dotenv import load_dotenv
load_dotenv()

SUBTITLE_PATHS = {
    'Entire Series': r'Data\HunterxHunterSubtitles',
    'Hunter Exam Arc': r'Data\HunterxHunterSubtitles\Hunter Exam Arc',
    'Zoldyck Family Arc': r'Data\HunterxHunterSubtitles\Zoldyck Family Arc',
    'Heavens Arena Arc': r'Data\HunterxHunterSubtitles\Heavens Arena Arc',
    'Yorknew City Arc': r'Data\HunterxHunterSubtitles\Yorknew City Arc',
    'Greed Island Arc': r'Data\HunterxHunterSubtitles\Greed Island Arc',
    'Chimera Ant Arc': r'Data\HunterxHunterSubtitles\Chimera Ant Arc',
    '13th Hunter Chairman Election Arc': r'Data\HunterxHunterSubtitles\13th Hunter Chairman Election Arc',
}

SUMMARIZER_OPTIONS = {
     'Concise Summary of Episode':(
        """
        Below is the full script from an episode of Hunter x Hunter. Write a single, well-structured paragraph that summarizes the entire episode.
        Focus on the key plot developments, character actions, and emotional turning points. Make sure the summary flows smoothly from beginning to end and captures the episode’s core story.
        Do not include unnecessary details, quotes, or dialogue formatting. Keep it concise and readable.
        {text}
        """
        ),
    'Bullet List of Events': (
        """
        Below is the full script fromm an episode of Hunter x Hunter. Write multiple bullet points on key events that occurred during the episode in chronological
        order. Focus on the key characters, actions, and settings of each event. Do not include unnecessary details, quotes, or dialogue formatting. Keep it concise and readable.
        {text} 
        """
        )
}

def get_themes(theme_list_str, selected_arc, save_path):
    subtitles_path = SUBTITLE_PATHS.get(selected_arc, None)
    if not subtitles_path or not os.path.exists(subtitles_path):
        return gr.HTML("<p style='color:red;'>Selected Arc's subtitles path not found. Please check configuration.</p>")
        
    theme_list = theme_list_str.split(',')
    theme_classifier = ThemeClassifier(theme_list)
    
    output_df, new_file = theme_classifier.get_themes(subtitles_path, save_path)
    
    if new_file:
        #Remove episode and script columns from output_df
        output_df = output_df[theme_list]
        output_df = output_df[theme_list].sum().reset_index()
    else:
        output_df = output_df.drop(columns = ['episode', 'script'])
        columns_list = list(output_df.columns)
        output_df = output_df[columns_list].sum().reset_index()
                
    output_df.columns = ['Theme', 'Score']
    
    output_chart = gr.BarPlot(
        output_df,
        x = 'Theme',
        y = 'Score',
        title = f'Classified Themes',
        tooltip = ['Theme', 'Score'],
        vertical = False,
        width = 500,
        height = 260
    )
    
    return output_chart

def get_summary(selected_output, episode_path, episode_save_path):
    prompt = SUMMARIZER_OPTIONS.get(selected_output)
    
    episode_summarizer = EpisodeSummarizer(prompt)
    
    episode_summary = episode_summarizer.get_episode_summary(episode_path, episode_save_path)
    
    return episode_summary

def get_character_network(selected_arc, ner_path):
    subtitles_path = SUBTITLE_PATHS.get(selected_arc)
    ner = NamedEntityRecognizer()
    ner_df = ner.get_ners(subtitles_path, ner_path)
        
    character_network_generator = CharacterNetworkGenerator()
    
    relationship_df = character_network_generator.generate_character_network(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)
    
    return html
    
def classify_text(text_classification_model, text_classification_data_path, text_to_classify):
    nen_classifier =  NenClassifier(model_path = text_classification_model, data_path = text_classification_data_path,
                                       huggingface_token = os.getenv('huggingface_token'))
    
    output = nen_classifier.classify_nen(text_to_classify)
    
    return output

def chat_with_character_chatbot(message, history):
    character_chatbot = CharacterChatBot('athynne/Naruto_Llama-3-8B', huggingface_token = os.getenv('huggingface_token'))
    
    output = character_chatbot.chat(message, history)
    output = output['content'].strip()
    
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
                        dropdown_menu = gr.Dropdown(
                            choices = list(SUBTITLE_PATHS.keys()), label = 'Arc', 
                            allow_custom_value = False, filterable = False, value = 'Entire Series'
                        )
                        save_path = gr.Textbox(label = "Save Path")
                        get_themes_button = gr.Button("Get Themes")
                        get_themes_button.click(get_themes, inputs = [theme_list, dropdown_menu, save_path], outputs = [plot])
                        
        #Episode Summarizer area
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Episode Summarizer with LLMs</h1>")
                with gr.Row():
                    with gr.Column():
                        episode_summarizer_output = gr.Textbox(label = "Episode Summarizer Output")
                    with gr.Column():
                        summarizer_dropdown_menu = gr.Dropdown(
                            choices = list(SUMMARIZER_OPTIONS.keys()), label = 'Type of Output', 
                            allow_custom_value = False, filterable = False, value = 'Concise Summary of Episode'
                        )
                        episode_to_summarize_path = gr.Textbox(label = 'Episode Path')
                        episode_summarization_save_path = gr.Textbox(label = 'Save Path')
                        get_summary_button = gr.Button('Get Summary')
                        get_summary_button.click(get_summary, inputs = [summarizer_dropdown_menu, episode_to_summarize_path, episode_summarization_save_path], outputs = [episode_summarizer_output])
    
        #Text Classification area
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Text Classification with LLMs</h1>")
                with gr.Row():
                    with gr.Column():
                        text_classification_output = gr.Textbox(label = 'Text Classification Output')
                    with gr.Column():
                        text_classification_model = gr.Textbox(label = 'Model Path')
                        text_classifcation_data_path = gr.Textbox(label='Data Path')
                        text_to_classify = gr.Textbox(label = 'Text Input')
                        classify_text_button = gr.Button('Classify Text (Nen)')
                        classify_text_button.click(classify_text, inputs = [text_classification_model, text_classifcation_data_path, text_to_classify], outputs = [text_classification_output])
                        
        #Character Network Area
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Network(NERs and Graphs)</h1>")
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                    with gr.Column():
                        cna_dropdown_menu = gr.Dropdown(
                            choices = list(SUBTITLE_PATHS.keys()), label = 'Arc', 
                            allow_custom_value = False, filterable = False, value = 'Entire Series'
                        )
                        ner_path = gr.Textbox(label = "NERs Save Path")
                        get_network_graph_button = gr.Button("Get Character Network")
                        get_network_graph_button.click(get_character_network, inputs = [cna_dropdown_menu, ner_path], outputs = [network_html])
                        
        #Character chatbot section
        with gr.Row():
            with gr.Column():
                gr.HTML('<h1>Character Chatbot</h1>')
                gr.ChatInterface(chat_with_character_chatbot)
                            
    iface.launch(share = True)
                
    
if __name__ == '__main__':
    main()
