import gradio as gr

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
                        subtitles_path = gr.Textbox(label = "Subtitles or script Path")
                        save_path = gr.Textbox(label = "Save Path")
                        get_themes_button = gr.Button("Get Themes")
                    
    iface.launch(share = True)
                
    
if __name__ == '__main__':
    main()
