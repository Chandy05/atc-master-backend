from pinecone import Pinecone, ServerlessSpec
import time

# 🔑 ใส่ API Key ของ Pinecone ตรงนี้
PINECONE_API_KEY = "pcsk_65kXi2_HoAgQRLJ9CqckM21zeCaBKGs3rRcno7dDAdgQNRpBiKtKu4fx6e3hePEQ7xvK5R"
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "atc-master"

print(f"🏗️ กำลังสร้างกล่องใหม่ ชื่อ '{index_name}' ขนาด 768 มิติ...")
pc.create_index(
    name=index_name,
    dimension=768, 
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

print("⏳ รอให้ระบบ Pinecone เซ็ตอัปตัวเองแป๊บนึง (ประมาณ 10 วินาที)...")
time.sleep(10)
print("✅ สร้างกล่องสำเร็จเรียบร้อยแล้ว! พร้อมอัปโหลดข้อมูลครับ")