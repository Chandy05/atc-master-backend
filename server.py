from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import re
import fitz  # PyMuPDF
import docx2txt
import base64
import uuid 
import json
import time

# 🌟 ใช้ Google GenAI SDK รุ่นใหม่ล่าสุด
from google import genai
from google.genai import types

from pinecone import Pinecone
from openai import OpenAI

# 🔑 ใส่ API Keys 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "ใส่ Key เดิมสำรองไว้เผื่อรันในเครื่อง")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ใส่ Key เดิมสำรองไว้เผื่อรันในเครื่อง")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "ใส่ Key เดิมสำรองไว้เผื่อรันในเครื่อง")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel): message: str; history: List[Dict[str, str]] = []; category: str; model: str
class SolveChunkRequest(BaseModel): session_id: str; chunk_index: int; category: str; model: str

print("กำลังเตรียมความพร้อมระบบ (V.6.2 Bring back exact Paragraph)...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("atc-master")
client = OpenAI(api_key=OPENAI_API_KEY)

EXAM_SCOPE = "1. Rules of the Air 2. Aerodrome Control 3. Approach & Area Control 4. AIS 5. Communication 6. Meteorology 7. Navigation 8. Aerodrome Facilities 9. RTAF"

exam_sessions = {}

@app.get("/")
def serve_frontend(): return FileResponse("index.html")

@app.get("/health")
def wake_up(): return {"status": "ตื่นแล้วจ้า!"}

@app.get("/categories")
def get_categories():
    try:
        stats = index.describe_index_stats()
        namespaces = list(set(stats.get("namespaces", {}).keys()))
        if not namespaces: namespaces = ["General"]
        return {"status": "success", "categories": namespaces}
    except Exception: 
        return {"status": "success", "categories": ["General"]}

def get_context_from_db(query_text, exact_keywords, category, top_k=25): 
    res = client.embeddings.create(input=[query_text], model="text-embedding-3-small", dimensions=768)
    query_vector = res.data[0].embedding
    all_matches = []
    fetch_amount = 40 
    
    if category == "ALL":
        namespaces = list(index.describe_index_stats().get("namespaces", {}).keys())
        if not namespaces: namespaces = [""]
        for ns in namespaces:
            try:
                res_query = index.query(vector=query_vector, top_k=fetch_amount, namespace=ns, include_metadata=True)
                all_matches.extend(res_query.get('matches', []))
            except: pass
        all_matches.sort(key=lambda x: x.get('score', 0), reverse=True)
    else:
        res_query = index.query(vector=query_vector, top_k=fetch_amount, namespace=category, include_metadata=True)
        all_matches = res_query.get('matches', [])

    exact_list = []
    normal_list = []
    kw_list = [k.strip().lower() for k in exact_keywords.split() if len(k.strip()) > 1]
    
    for m in all_matches:
        text_lower = m['metadata'].get('text', '').lower()
        if any(kw in text_lower for kw in kw_list): exact_list.append(m)
        else: normal_list.append(m)
            
    final_matches = (exact_list + normal_list)[:top_k]
    return "\n\n".join([f"Source: [{m['metadata'].get('source', 'Unknown')}]\nText: {m['metadata'].get('text', '')}" for m in final_matches if 'metadata' in m])

@app.post("/upload_exam")
async def upload_exam_api(files: List[UploadFile] = File(...)):
    try:
        extracted_text = ""
        for file in files:
            ext = file.filename.split('.')[-1].lower()
            content = await file.read()
            if ext in ['pdf']:
                doc = fitz.open(stream=content, filetype="pdf")
                for page in doc: extracted_text += page.get_text() + "\n"
            elif ext in ['doc', 'docx']:
                with open("temp.docx", "wb") as f: f.write(content)
                extracted_text += docx2txt.process("temp.docx") + "\n"
                os.remove("temp.docx")
            elif ext in ['jpg', 'jpeg', 'png']:
                mime_type = "image/png" if ext == "png" else "image/jpeg"
                base64_img = base64.b64encode(content).decode('utf-8')
                ocr_res = client.chat.completions.create(
                    model="gpt-4o", 
                    messages=[
                        {"role": "system", "content": "Extract all text exactly as written."},
                        {"role": "user", "content": [{"type": "text", "text": "Read this image."}, {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_img}"}}]}
                    ], max_tokens=1500
                )
                extracted_text += ocr_res.choices[0].message.content + "\n\n"

        extracted_text = re.sub(r'(?i)^.*page\s+\d+\s+of\s+\d+.*$', '', extracted_text, flags=re.MULTILINE)
        
        split_prompt = f"""
        Analyze the following text which contains an exam.
        Separate it into individual questions (including their choices).
        Pay STRICT ATTENTION to the question numbers in the text. 
        If a question number is unreadable or missing, you MUST mathematically infer it based on the previous question (e.g., if you see Q13, then an unclear question, then Q15, the unclear one MUST be numbered 14).
        
        Return ONLY a valid JSON array of objects. Format:
        [
            {{"q_num": 13, "q_text": "13. What is...? a. x b. y c. z"}},
            {{"q_num": 14, "q_text": "How to...? a. x b. y c. z"}} 
        ]
        
        Text to split:
        {extracted_text}
        """
        
        split_res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": split_prompt}],
            max_tokens=4000
        )
        
        raw_json = split_res.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        
        try:
            final_chunks = json.loads(raw_json)
        except:
            raw_chunks = re.split(r'(?=\n\d+\.)', "\n" + extracted_text)
            final_chunks = [{"q_num": i+1, "q_text": c.strip()} for i, c in enumerate(raw_chunks) if c.strip() and len(c.strip()) > 10]

        if not final_chunks: final_chunks = [{"q_num": 1, "q_text": extracted_text}]

        session_id = str(uuid.uuid4())
        exam_sessions[session_id] = final_chunks
        return {"status": "success", "session_id": session_id, "total_chunks": len(final_chunks)}
    except Exception as e: return {"status": "error", "message": str(e)}

@app.post("/solve_chunk")
def solve_chunk_api(req: SolveChunkRequest):
    try:
        time.sleep(1) 
        if req.session_id not in exam_sessions: return {"status": "error", "message": "Session Expired"}
        
        chunk_data = exam_sessions[req.session_id][req.chunk_index]
        q_num = chunk_data.get("q_num", req.chunk_index + 1)
        chunk_text = chunk_data.get("q_text", "")
        
        if req.category == "ENG":
            context = "ไม่มีเอกสารอ้างอิง เนื่องจากเป็นข้อสอบวิชาภาษาอังกฤษทั่วไป"
            category_label = "ENG"
            system_role = "คุณคือสุดยอดอาจารย์สอนภาษาอังกฤษ (English Tutor) ผู้เชี่ยวชาญด้านไวยากรณ์ คำศัพท์ และข้อสอบ TOEIC"
            step2_instruction = "- Step 2: วิเคราะห์หลักไวยากรณ์/คำศัพท์ = \"[Explain the grammar rules, vocabulary meaning, or syntax structure related to the question]\""
            source_instruction = "MUST write \"( ENG ) 🧠 หลักไวยากรณ์ภาษาอังกฤษสากล\"."
            kw_tokens = 0
            
        else:
            # โหมด ATC ปกติ
            kw_prompt = f"Extract only technical codes, acronyms, or specific nouns from this single question. DO NOT add any new words. If none, return nothing.\n\nText: {chunk_text}"
            kw_res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": kw_prompt}], max_tokens=30)
            exact_keywords = kw_res.choices[0].message.content.strip()
            kw_tokens = kw_res.usage.prompt_tokens + kw_res.usage.completion_tokens
            
            context = get_context_from_db(chunk_text, exact_keywords, req.category, top_k=20)
            category_label = req.category if req.category != "ALL" else "ทุกหมวดหมู่"
            system_role = "You are a highly accurate Aviation ATC Exam Solver. You never guess, you only analyze step-by-step."
            step2_instruction = "- Step 2: วิเคราะห์ตามเอกสารอ้างอิง = \"[Explain what the provided Document Context says about this specific topic]\""
            
            # 🌟 ล็อกคอ AI เรื่องแหล่งอ้างอิง (บังคับเอาพารากราฟกลับมา แต่ห้ามมโน)
            source_instruction = f"MUST start with \"( {category_label} ) \". You MUST write the EXACT File Name from the Context. IF the text contains a specific Chapter, Section, or Paragraph number, you MUST include it (e.g., '( OJT ) Source: doc.pdf, Chapter 3, Para 1.2'). DO NOT invent fake numbers. If NOT found in Context -> Score 50-89 and write \"( {category_label} ) 🧠 ความรู้สากลออนไลน์\"."

        # 🌟 PROMPT หลักที่ปรับปรุง
        final_prompt = f"""[Document Context]:\n{context}\n
[Exam Question]:\n{chunk_text}\n
TASK: Solve the single question provided above.

CRITICAL INSTRUCTIONS (MUST OBEY):
1. "คำถาม" and "ช้อยส์": Write exactly as given in the text.
2. "คำตอบที่ถูกต้อง": You MUST write the full letter AND the full exact text of the correct choice.
3. "อธิบาย": ACT AS A LOGICAL ANALYST. You MUST use this exact 3-step format in Thai:
   - Step 1: คำถามนี้ ถามว่า .... = "[Translate the question to Thai. If already in Thai, just restate it clearly]"
   {step2_instruction}
   - Step 3: สรุปเหตุผล = "[Explain logically WHY the correct choice matches the rule and WHY others are wrong]"
4. "ความน่าเชื่อถือ": MUST BE EXACTLY ONE INTEGER NUMBER between 0 and 100.
5. "แหล่งอ้างอิง": {source_instruction}
6. "ข้อความต้นฉบับ": Quote the relevant sentence from the Context. THEN, you MUST find the specific keyword that proves the answer and wrap it EXACTLY in this HTML tag to make it black and bold: <b><span style="color: #000000; font-weight: 900;">[KEYWORD]</span></b>.

Generate ONLY valid HTML (replace bracket placeholders with actual data):
<div id="q-{q_num}" class="question-container" style="margin-bottom: 30px; page-break-inside: avoid; position: relative;">
    <h4 class="q-num">📌 ข้อที่ {q_num}</h4>
    <p class="q-text"><b>📝 คำถาม:</b><br>[Question text]<br>[Choice A]<br>[Choice B]<br>[Choice C]<br>[Choice D]</p>
    <p class="a-text"><b>🎯 คำตอบที่ถูกต้อง:</b> <span style="color: #27ae60; font-weight: bold;">[Correct Answer Letter AND Full Text]</span></p>
    <p class="e-text" style="line-height: 1.8;"><b>💡 อธิบาย:</b><br>
    <b>- Step 1: คำถามนี้ ถามว่า .... =</b> "[Translation/Restatement]"<br>
    <b>{step2_instruction.split('=')[0].strip()} =</b> "[Analysis]"<br>
    <b>- Step 3: สรุปเหตุผล =</b> "[Final Conclusion]"</p>
    <p class="c-text"><b>📊 ความน่าเชื่อถือ:</b> [INTEGER SCORE]%</p>
    <div style="background: #f8fafc; padding: 15px; border-left: 4px solid #cbd5e1; margin-top: 10px;">
        <b class="ref-text">📚 แหล่งอ้างอิง:</b> [Source Name AND Paragraph/Section if available]<br>
        <b>🔎 ข้อความต้นฉบับ:</b> "[Quote with <b><span style='color: #000000; font-weight: 900;'>ANSWER KEYWORD</span></b> highlighted]"
    </div>
    <div class="mt-3 text-end d-print-none no-export">
        <button class="btn btn-sm btn-info text-white shadow-sm" onclick="requestInfographic(this)"><i class="bi bi-easel-fill"></i> สรุปเป็น Infographic ภาพ</button>
    </div>
    <script class="no-export">
        setTimeout(() => {{ try {{ addATCStrip(`q-{q_num}`, `[Short Question Text max 30 chars]...`, `[INTEGER SCORE]%`); }} catch(e){{}} }}, 500);
    </script>
</div><hr style="border-top: 2px dashed #ccc;" class="no-export">"""

        response = client.chat.completions.create(
            model=req.model, 
            messages=[{"role": "system", "content": system_role}, {"role": "user", "content": final_prompt}], 
            max_tokens=2500
        )
        content = response.choices[0].message.content.replace('```html', '').replace('```', '')
        
        total_input = response.usage.prompt_tokens + kw_tokens
        total_output = response.usage.completion_tokens
        
        return {"status": "success", "answer": content, "tokens": {"input": total_input, "output": total_output}}
    except Exception as e: return {"status": "error", "message": str(e)}

@app.post("/chat")
def chat_api(req: ChatRequest):
    try:
        if req.category == "ENG":
            context = "นี่คือการสนทนาวิชาภาษาอังกฤษทั่วไป"
            persona = "คุณคืออาจารย์สอนภาษาอังกฤษ (English Tutor) ตอบคำถามเรื่องไวยากรณ์และ TOEIC"
        else:
            kw_prompt = f"Extract ONLY the main nouns, specific codes, or technical terms from this user query: '{req.message}'. DO NOT add any extra words. If none, output nothing."
            kw_response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": kw_prompt}], max_tokens=30)
            exact_keywords = kw_response.choices[0].message.content.strip()
            context = get_context_from_db(req.message, exact_keywords, req.category, top_k=20)
            persona = "คุณคือ AI ติวเตอร์และผู้เชี่ยวชาญด้านการบิน (ATC) ของไทย"
            
        gemini_prompt = f"""{persona}
[ข้อมูลอ้างอิงจากฐานข้อมูลของคุณ]:\n{context}\n
กฎกติกา:
1. อ่านและทำความเข้าใจคำถามจากผู้ใช้ให้ดี 
2. ใช้ข้อมูลอ้างอิงด้านบนประกอบการตอบ หากคำถามคือการถามหารหัส ให้ดูข้อมูลจากอ้างอิงเป็นหลัก
3. ถ้าข้อมูลในอ้างอิงไม่ชัดเจน คุณสามารถใช้ความรู้สากลของคุณ ในการอธิบายเพิ่มเติมได้เลย

คำถามของผู้ใช้: {req.message}
"""
        gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
        
        response = gemini_client.models.generate_content(
            model='gemini-1.5-flash',
            contents=gemini_prompt,
            config=types.GenerateContentConfig(
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                ]
            )
        )
        return {"status": "success", "answer": response.text, "tokens": {"input": 0, "output": 0}} 
    except Exception as e: 
        return {"status": "error", "message": str(e)}