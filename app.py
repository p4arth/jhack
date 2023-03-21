import gradio as gr
import requests
import cohere
from translate import Translator

def get_weather_data():
    lat = 27.0238
    lon = 74.2179
    key = "d07f2290a30824b709b50a94237cfcb7"
    url = f"https://api.openweathermap.org/data/2.5/weather?units=metric&lat={lat}&lon={lon}&appid={key}"
    response = requests.get(url)
    result = response.json()
    result["main"].update({"description": result["weather"][0]["description"]})
    return result["main"]

def generate_prompt(data):
    weather_json = get_weather_data()
    prompt = \
    f'''State: Rajasthan
Max Temprature: {weather_json["temp_max"]}
Min Temprature: {weather_json["temp_min"]}
Humidity: {weather_json["humidity"]}
Weather Description: {weather_json["description"]}
Context: Through increased use of soil testing and plant analyses, micronutrient deficiencies have been verified in many soils. Some reasons limiting the incidental additions of micronutrients include.High-yield crop demands remove micronutrients from the soil. Increased use of high-analysis NPK fertilizers containing lower quantities of micronutrient contaminants. Advances in fertilizer technology reduce the residual addition of micronutrients.
Question: {data}
'''
    return prompt

def get_response(prompt):
    co = cohere.Client('EoYqxEa60C0EEeKadblGW8NE94geVCEE75lDqySe')
    new_prompt = generate_prompt(prompt)
    response = co.generate(
        model='command-xlarge-nightly',  
        prompt = new_prompt,  
        max_tokens = 1000,
        temperature = 0.6,  
        stop_sequences = ["--"]
    )
    translator = Translator(to_lang="hi")
    translation = translator.translate(response.generations[0].text)
    return translation

title = """<h1 align="center">🌱 Farmer Queries LLM | किसान प्रश्न LLM 🌾</h1>"""
with gr.Blocks(css="""#col_container {margin-left: auto; margin-right: auto;} #chatbot {height: 520px; overflow: auto;}""") as demo:
    gr.HTML(title)
    input1 = gr.Textbox(label = 'प्रश्न पूछो')
    output1 = gr.Textbox(label = 'उत्तर')
    btn = gr.Button("जमा करे")
    btn.click(get_response, [input1], output1)
demo.launch()