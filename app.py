#Load imports
from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

#Folder to save uploaded files
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model('model/resnet50V2_model1.h5')

# class names
class_names = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox']

# Ensure folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('index2.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)  # Saves the file to the uploads folder
        
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=(244, 244))
        img_array = img_to_array(img) / 255.0  
        img_array = np.expand_dims(img_array, axis=0) 
        
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100 

        #Text of presdicted class
        if predicted_class == "Monkeypox":
            description = """ Mpox  or Monkeypox is an infectious disease that can cause a painful rash, enlarged lymph nodes, fever, headache, muscle ache, back pain and low energy. Most people fully recover, but some get very sick.
            
    Mpox is caused by the monkeypox virus (MPXV). It is an enveloped double-stranded DNA virus of the Orthopoxvirus genus in the Poxviridae family, which includes variola, cowpox, vaccinia and other viruses. There are two distinct clades of the virus: clade I (with subclades Ia and Ib) and clade II (with subclades IIa and IIb).

    Mpox spreads from person to person mainly through close contact with someone who has mpox, including members of a household. Close contact includes skin-to-skin (such as touching or sex) and mouth-to-mouth or mouth-to-skin contact (such as kissing), and it can also include being face-to-face with someone who has mpox (such as talking or breathing close to one another, which can generate infectious respiratory particles).
            
    Mpox causes signs and symptoms which usually begin within a week but can start 1–21 days after exposure. Symptoms typically last 2–4 weeks but may last longer in someone with a weakened immune system. 

            Common symptoms of mpox are: 
            rash 
            fever 
            sore throat 
            headache 
            muscle aches 
            back pain 
            low energy 
            swollen lymph nodes.
            
    Identifying mpox can be difficult because other infections and conditions can look similar. It is important to distinguish mpox from chickenpox, measles, bacterial skin infections, scabies, herpes, syphilis, other sexually transmitted infections, and medication-associated allergies. Someone with mpox may also have another sexually transmitted infection at the same time, such as syphilis or herpes. Alternatively, a child with suspected mpox may also have chickenpox. For these reasons, testing is key for people to get care as early as possible and prevent severe illness and further spread.
            
    The goal of treating mpox is to take care of the rash, manage pain and prevent complications. Early and supportive care is important to help manage symptoms and avoid further problems. 

    Getting an mpox vaccine can help prevent infection (pre-exposure prophylaxis). It is recommended for people at high-risk of getting mpox, especially during an outbreak."""
        elif predicted_class == 'Measles':
            description = """Measles is a highly contagious disease caused by a virus. It spreads easily when an infected person breathes, coughs or sneezes. It can cause severe disease, complications, and even death. Measles can affect anyone but is most common in children.   

    Measles infects the respiratory tract and then spreads throughout the body. Symptoms include a high fever, cough, runny nose and a rash all over the body.
            
    Measles is still common, particularly in parts of Africa, the Middle East and Asia. The overwhelming majority of measles deaths occur in countries with low per capita incomes or weak health infrastructures that struggle to reach all children with immunization.
            
    All children or adults with measles should receive two doses of vitamin A supplements, given 24 hours apart. This restores low vitamin A levels that occur even in well-nourished children. It can help prevent eye damage and blindness. Vitamin A supplements may also reduce the number of measles deaths."""
        elif predicted_class == 'Chickenpox':
            description = """Varicella (chickenpox) is an acute, highly contagious disease caused by varicella-zoster virus (VZV), a member of the herpesvirus family. Only one serotype of VZV is known, and humans are the only reservoir. Following infection, the virus remains latent in neural ganglia and in about 10-20% of cases it is reactivated it is reactivated to cause herpes zoster, or shingles, generally in persons over 50 years of age or immunocompromised individuals.

    VZV transmission occurs via droplets, aerosols, or direct contact with respiratory secretions, and almost always produces clinical disease in susceptible individuals. While mostly a mild disorder in childhood, varicella tends to be more severe in adults. It may be fatal, especially in neonates and in immunocompromised persons. In temperate climates most cases occur before the age of 10. Varicella is characterized by an itchy, rash usually starting on the scalp and face and initially accompanied by fever and malaise. The rash gradually spreads to the trunk and extremities. The vesicles gradually dry out and crusts appear which then disappear over a period of one to two weeks.

    The infection may occasionally be complicated by pneumonia or encephalitis (inflammation of the brain), at times with serious or fatal consequences. Shingles is a painful rash that may occasionally result in permanent damage to the nerves or visual impairment. It is relatively common in HIV-infected persons, sometimes with fatal consequences."""
        elif predicted_class == 'Cowpox':
            description = """Cowpox, uncommon mildly eruptive disease of animals, first observed in cows and occurring particularly in cats, that when transmitted to otherwise healthy humans produces immunity to smallpox. The cowpox virus is closely related to variola, the causative virus of smallpox. The word vaccinia is sometimes used interchangeably with cowpox to refer to the human form of the disease, sometimes to refer to the causative virus, and sometimes to refer only to the artificially induced human form of cowpox.

    Most human cases of cowpox appear as one or a small number of pus-like lesions on the hands and face, which then ulcerate and form a black scab before healing on their own. This process can take up to 12 weeks, with the following skin findings over that period:

        Days 1–6 (after infection): the site of infection appears as an inflamed macule.
        Days 7–12: the inflamed lesion becomes raised (a papule), then develops into a vesicle.

        Days 13–20: the vesicle becomes filled with blood and pus and eventually ulcerates. Other lesions may develop close by.
        Weeks 3–6: the ulcerated wound turns into a deep-seated, hard, black crusty eschar, which is surrounded by redness and swelling.
        Weeks 6–12: the eschar begins to flake and slough and the lesion heals, often leaving a scar.
    
    There is no cure for cowpox, but the disease is self-limiting. The human immune response is sufficient to control the infections on its own. The lesions heal by themselves within 6–12 weeks. Often patients are left with scars at the site of the healed pox lesions"""
        elif predicted_class == 'HFMD':
            description = """Hand-foot-and-mouth disease is a mild, contagious viral infection common in young children. Symptoms include sores in the mouth and a rash on the hands and feet. Hand-foot-and-mouth disease is most commonly caused by a coxsackievirus.

    Hand-foot-and-mouth disease may cause all of the following symptoms or only some of them. They include:

        Fever.
        Sore throat.
        Feeling sick.
        Painful, blister-like lesions on the tongue, gums and inside of the cheeks.
        A rash on the palms, soles and sometimes the buttocks. The rash is not itchy, but sometimes it has blisters. Depending on skin tone, the rash may appear red, white, gray, or only show as tiny bumps.
        Fussiness in infants and toddlers.
        Loss of appetite.
    
    The most common complication of hand-foot-and-mouth disease is dehydration. The illness can cause sores in the mouth and throat, making it painful to swallow. Encourage your child to drink fluids during the illness. If children become too dehydrated, they may need intravenous (IV) fluids in the hospital. Hand-foot-and-mouth disease is usually a minor illness. It usually only causes fever and mild symptoms for a few days."""
        elif predicted_class == 'Healthy':
            description = """This indicates that the skin uploaded is healthy"""
        else:
            description = "No description available for this condition."

        
        return render_template('display.html', image_url=file_path, prediction=predicted_class, description=description, confidence=confidence)

@app.route('/display/<filename>')
def display(filename):
    # Use url_for to generate the file path
    file_url = url_for('static', filename=f'uploads/{filename}')
    return render_template('display.html', image_url=file_url)

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)