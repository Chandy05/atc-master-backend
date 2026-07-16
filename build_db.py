import os 
import glob 
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma 
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
import time 
 
# 🔑 ใส่ API KEY ของคุณที่นี่ 
os.environ["GOOGLE_API_KEY"] = "XXX" 
 
# ตั้งค่าที่อยู่ของไฟล์ PDF (อ้างอิงจากรูปภาพของคุณ) 
PDF_FOLDER_PATH = r"D:\01 Project\Atc_Essential" 
DB_DIR = "./chroma_db" 
 
def build_vector_database(): 
    print(f"🔍 กำลังสแกนหาไฟล์ PDF ใน: {PDF_FOLDER_PATH}") 
    
    # ค้นหาไฟล์ PDF ทั้งหมดในโฟลเดอร์หลักและโฟลเดอร์ย่อย 
    pdf_files = glob.glob(os.path.join(PDF_FOLDER_PATH, '**', '*.pdf'), recursive=True) 
    
    if not pdf_files: 
        print("❌ ไม่พบไฟล์ PDF ในโฟลเดอร์ที่ระบุ") 
        return 
 
    print(f"📚 พบไฟล์ PDF ทั้งหมด: {len(pdf_files)} ไฟล์") 
    
    # ใช้ Embedding Model ของ Google (ฟรีและรองรับภาษาไทย/อังกฤษได้ดี) 
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
    
    # เครื่องมือหั่นข้อความ (Chunking) 
    text_splitter = RecursiveCharacterTextSplitter( 
        chunk_size=1000, 
        chunk_overlap=200 
    ) 
 
    # วนลูปอ่านและบันทึกทีละไฟล์ (เพื่อป้องกัน RAM เต็มเพราะไฟล์มีขนาด 1.6GB) 
    for i, file_path in enumerate(pdf_files, 1): 
        try: 
            print(f"⏳ [{i}/{len(pdf_files)}] กำลังประมวลผล: {os.path.basename(file_path)}...") 
            
            loader = PyPDFLoader(file_path) 
            docs = loader.load() 
            splits = text_splitter.split_documents(docs) 
            
            # บันทึกลง ChromaDB (จะสร้างโฟลเดอร์ chroma_db ให้อัตโนมัติ) 
            Chroma.from_documents( 
                documents=splits, 
                embedding=embeddings, 
                persist_directory=DB_DIR 
            ) 
            
            # หน่วงเวลาเล็กน้อย ป้องกัน API Limit ของ Google 
            time.sleep(1) 
            
        except Exception as e: 
            print(f"⚠️ ข้ามไฟล์ {os.path.basename(file_path)} เนื่องจากพบข้อผิดพลาด: {e}") 
 
    print("✅ สร้างฐานข้อมูล Vector สำเร็จเรียบร้อยแล้ว! (ข้อมูลถูกเก็บไว้ในโฟลเดอร์ chroma_db)") 
 
if __name__ == "__main__": 
    build_vector_database()
