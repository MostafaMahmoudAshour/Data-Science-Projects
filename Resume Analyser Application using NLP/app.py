import streamlit as st
import pickle
import nltk
import re

nltk.download('punkt')
nltk.download('stopwords')

# loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def clean_resume(text):
    cleaned_text = re.sub('http\S+\s', ' ', text)
    cleaned_text = re.sub('@\S+', ' ', cleaned_text)
    cleaned_text = re.sub('#\S+\s', ' ', cleaned_text)
    cleaned_text = re.sub('RT|CC', ' ', cleaned_text)
    cleaned_text = re.sub('[%s]' % re.escape("""!#$%^&*_.-+~`"<=>/\|[]{};:@()?',"""), ' ', cleaned_text)
    cleaned_text = re.sub(r'[^\x00-\x7f]', ' ', cleaned_text) # replace Emojis and matches any character not in the ASCII rang
    cleaned_text = re.sub('\s+', ' ', cleaned_text)
    return cleaned_text

# web app
def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])
    
    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # if UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        
        # Map category ID to Category Name
        category_mapping = {
            6: 'Data Science', 
            12: 'HR', 
            0: 'Advocate', 
            1: 'Arts', 
            24: 'Web Designing',
            16: 'Mechanical Engineer',
            22: 'Sales',
            14: 'Health and fitness',
            5: 'Civil Engineer',
            15: 'Java Developer',
            4: 'Business Analyst',
            21: 'SAP Developer',
            2: 'Automation Testing',
            11: 'Electrical Engineering',
            18: 'Operations Manager',
            20: 'Python Developer',
            8: 'DevOps Engineer',
            17: 'Network Security Engineer',
            19: 'PMO',
            7: 'Database',
            13: 'Hadoop',
            10: 'ETL Developer',
            9: 'DotNet Developer',
            3: 'Blockchain',
            23: 'Testing'
        }
        
        category_name = category_mapping.get(prediction_id, 'Unknown')
        st.write('Predicted Category: ', category_name)

# python main
if __name__ == "__main__":
    main()