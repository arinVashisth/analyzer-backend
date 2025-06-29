from flask import Flask, request, jsonify
from flask_cors import CORS
from resume_parser import extract_resume_text
from job_matcher import match_resume_with_jd
from dotenv import load_dotenv
from openai import OpenAI
import os


app = Flask(__name__)
CORS(app)



@app.route('/analyze', methods=['POST'])
def analyze_resume():
    try:
        resume_file = request.files['resume']
        jd_text = request.form['job_description']
        resume_text = extract_resume_text(resume_file)
        score, matching_skills, missing_skills,name,email,links = match_resume_with_jd(resume_text, jd_text)
        match_score = score
        if match_score >= 80:
            suggestions = "Your resume is a strong match! You’re good to go 🎯."
        elif match_score >= 60:
            if missing_skills:
                suggestions = f"You're a good match, but consider learning: {', '.join(missing_skills[:5])}."
            else:
                suggestions = "You're a good match. Just polish your resume a bit more!"
        elif match_score >= 40:
            if missing_skills:
                suggestions = f"Your resume needs improvement. Learn these skills: {', '.join(missing_skills[:5])}."
            else:
                suggestions = "Improve your resume formatting or add more relevant projects."
        else:
            if missing_skills:
                suggestions = f"Significant skill gaps found. Start with: {', '.join(missing_skills[:5])}."
            else:
                suggestions = "Consider rewriting your resume to align better with the job."


        return jsonify({
            'resume_text': resume_text,
            'match_score': float(score),
            'matching_skills': list(matching_skills),
            'missing_skills': list(missing_skills),
            'gpt_suggestions': suggestions,
            'name': name,
            'email': email,
            'links': links
        })


    except Exception as e:
        print("Resume Analysis Error:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)