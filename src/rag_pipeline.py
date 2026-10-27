#PDF DOSYALARINI YÜKLEME

# %%  
import os

from pathlib import Path
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:  # VS Code interaktif penceresinde __file__ olmaz
    PROJECT_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
knowledge_base_path = str(PROJECT_ROOT / "data")

def create_knowledge_base(knowledge_base_path):

    downloaded_files = []

    for item in os.listdir(knowledge_base_path):

        item_path = os.path.join(knowledge_base_path, item)

        if os.path.isfile(item_path) and item.lower().endswith(".pdf"):
            downloaded_files.append(item)

    print(f"'{knowledge_base_path}' dizininde bulunan PDF dosyaları:")

    for file_path in downloaded_files:
        print(f"📄 {file_path}")

    print(f"\nToplam {len(downloaded_files)} PDF dosyası bulundu.")

    return downloaded_files


knowledge_base = create_knowledge_base(knowledge_base_path)

# %%”
import PyPDF2
def convert_PDF_Text(knowledge_base_path, file_name):
  file_path= os.path.join(knowledge_base_path, file_name)
  try:
    # Belirtilen PDF dosyasını okumak için PdfReader nesnesi oluştur
    reader = PyPDF2.PdfReader(file_path)

    # Her sayfadan metni çıkar ve baş/son boşlukları temizle
    pdf_pages = [p.extract_text().strip() for p in reader.pages]
    # Boş metin dizilerini filtrele
    pdf_pages = [text for text in pdf_pages if text]
    # Belge adını ve işlenen sayfa sayısını yazdır
    print("📗", file_name,"yükleniyor...\nToplam ", len(pdf_pages), " Sayfa yüklendi ",)
    # Metin sayfalarının listesini döndür
    return pdf_pages
  except Exception as e:
    print(f"🚩Hata oluştu: {file_path} dosyası işlenemedi. Hata: {e}")
    return [] # Hata durumunda boş liste döndür


pdf_pages = []

for i in range(len(knowledge_base)):
    pdf_pages.extend(
        convert_PDF_Text(knowledge_base_path, knowledge_base[i])
    )

print("Toplam sayfa:", len(pdf_pages))


#CHUNKING İŞLEMİ
# %%

import textwrap
from IPython.display import display
from IPython.display import Markdown

def to_markdown(text):
  # Markdown formatına uygun hale getir, madde işaretlerini düzelt
  text = text.replace('•', '  *')
  # Metni girintile ve Markdown nesnesi olarak döndür
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
     
print(pdf_pages[0]) 

from langchain_text_splitters import RecursiveCharacterTextSplitter


def convert_Page_ChunkinChar(pdf_texts, chunk_size=1500, chunk_overlap=200):

    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    character_split_texts = character_splitter.split_text(
        '\n\n'.join(pdf_texts)
    )

    print(
        f"Toplam parça sayısı "
        f"(maksimum karakter boyutu = {chunk_size}): "
        f"{len(character_split_texts)}"
    )

    return character_split_texts

# %%
# Metinleri token bazlı parçalara ayırmak için gerekli sınıfı içe aktar
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

# Kullanılacak Sentence Transformer modelinin adını tanımla
# Bu model, metin parçalarını tokenlara ayırmak ve embedding işlemi için kullanılacak.
sentence_transformer_model="distiluse-base-multilingual-cased-v1"
     

# Metinleri token bazlı parçalara ayıran yardımcı fonksiyon
def convert_Chunk_Token(text_chunksinChar,
                        sentence_transformer_model,
                        chunk_overlap=10,
                        tokens_per_chunk=128 ):
  # SentenceTransformersTokenTextSplitter nesnesi oluşturuluyor
  token_splitter = SentenceTransformersTokenTextSplitter(
      chunk_overlap=chunk_overlap, # Parçalar arası çakışma miktarı
      model_name=sentence_transformer_model, # Kullanılacak tokenizer modelinin adı
      tokens_per_chunk=tokens_per_chunk) # Her bir parçadaki maksimum token sayısı

  # Tokenlara ayrılmış metin parçalarını tutacak boş liste
  text_chunksinTokens = []
  # Karakter bazlı parçalar üzerinde döngü
  for text in text_chunksinChar:
      # Her bir karakter bazlı parçayı tokenlara ayır ve listeye ekle
      text_chunksinTokens += token_splitter.split_text(text)
  # Bilgilendirme mesajı yazdır
  print(f"""\nBelge {tokens_per_chunk} token'lık parçalara bölündüğünde,
  ve çakışma token sayısı {chunk_overlap} olduğunda
  toplam parça sayısı: {len(text_chunksinTokens)}""")
  # Tokenlara ayrılmış metin parçaları listesini döndür
  return text_chunksinTokens
     

#vektör veri tabanı oluşturma
#%%
import chromadb
# Vektör veritabanı için dosya yolu belirle
vector_database_path = "./" # Mevcut dizini kullan

from chromadb.utils import embedding_functions
# SentenceTransformer modelini kullanarak bir embedding fonksiyonu oluştur
# Bu fonksiyon, metin parçalarını vektörlere dönüştürmek için kullanılacak

# önce embedding modelini seç
sentence_transformer_model="distiluse-base-multilingual-cased-v1"
# sonra fonksiyonu oluştur
embedding_function= embedding_functions.SentenceTransformerEmbeddingFunction(model_name=sentence_transformer_model)

def create_chroma_client(vector_database_path, collection_name, embedding_function):
  # Kalıcı (Persistent) bir ChromaDB istemcisi oluştur
  # Bu, vektör veritabanının diske kaydedilmesini ve oturumlar arasında kalıcı olmasını sağlar
  chroma_client = chromadb.PersistentClient(path=vector_database_path)

  # Belirtilen isimde bir koleksiyonun var olup olmadığını kontrol et
  try:
      chroma_collection = chroma_client.get_collection(collection_name)
      print(f"'{collection_name}' adlı koleksiyon zaten mevcut. Siliniyor...")
      # Eğer koleksiyon mevcutsa sil
      chroma_client.delete_collection(collection_name)
      print(f"'{collection_name}' adlı koleksiyon silindi.")
      # Silme işleminden sonra koleksiyonu yeniden oluştur
      chroma_collection = chroma_client.create_collection(collection_name, embedding_function=embedding_function)
      print(f"'{collection_name}' adlı koleksiyon yeniden oluşturuldu.")
  except:
      # Eğer koleksiyon mevcut değilse, doğrudan oluştur
      chroma_collection = chroma_client.create_collection(collection_name, embedding_function=embedding_function)
      print(f"'{collection_name}' adlı yeni koleksiyon oluşturuldu.")


  # Oluşturulan istemciyi ve koleksiyonu döndür
  return chroma_client, chroma_collection

def add_meta_data(text_chunksinTokens, title, category, initial_id):
  # Parçalar için benzersiz kimlikler (ID'ler) oluştur
  ids = [str(i+initial_id) for i in range(len(text_chunksinTokens))]
  # Parçalara eklenecek metadata (üst veri) sözlüğü oluştur
  metadata = {
      'document': title,  # Belge adı
      'category': category # Belge kategorisi
  }
  # Her parça için aynı metadata'yı içeren bir liste oluştur
  metadatas = [ metadata for i in range(len(text_chunksinTokens))]
  # Oluşturulan ID'ler ve metadata listelerini döndür
  return ids, metadatas
     

def add_document_to_collection(ids, metadatas, text_chunksinTokens, chroma_collection):
  # Ekleme işleminden önceki koleksiyon boyutunu yazdır
  print("Ekleme işleminden önceki koleksiyon boyutu: ", chroma_collection.count())
  # Belgeleri, metadata ve ID'leri belirterek ChromaDB koleksiyonuna ekle
  chroma_collection.add(ids=ids, metadatas= metadatas, documents=text_chunksinTokens)
  # Ekleme işleminden sonraki koleksiyon boyutunu yazdır
  print("Ekleme işleminden sonraki koleksiyon boyutu: ", chroma_collection.count())
  # Güncellenmiş koleksiyon nesnesini döndür
  return chroma_collection

# %%
# iş akışı
def load_multiple_pdfs_to_ChromaDB(collection_name,sentence_transformer_model):
  # Koleksiyon adını belirle
  collection_name= collection_name
  # Kategori adını belirle
  category= "Ogrenci Yonergeleri"
  # Kullanılacak Sentence Transformer modelini belirle
  sentence_transformer_model=sentence_transformer_model
  # Embedding fonksiyonunu oluştur
  embedding_function= embedding_functions.SentenceTransformerEmbeddingFunction(model_name=sentence_transformer_model)
  # ChromaDB istemcisi ve koleksiyonunu oluştur veya al
  chroma_client, chroma_collection = create_chroma_client(vector_database_path, collection_name, embedding_function)
  # Mevcut koleksiyonun boyutunu al (ID'leri başlatmak için)
  current_id = chroma_collection.count()
  # Bilgi tabanındaki dosya isimlerini al
  file_names = create_knowledge_base(knowledge_base_path)
  # Her dosya için döngü
  for file_name in file_names:
    # Belge işleme başlangıç mesajı
    print(f"\n📖 {file_name}, {chroma_collection.name} koleksiyonuna eklenmek üzere işleniyor. Kolleksiyondaki mevcut parça sayısı: {chroma_collection.count()}")
    print(f"\n📚 Kolleksiyondaki mevcut parça sayısı: {chroma_collection.count()}")
    # Başlangıç ID'sini yazdır
    print(f"\tBaşlangıç ID: {current_id} ")
    # PDF metnini sayfa sayfa çıkar
    pdf_texts = convert_PDF_Text(knowledge_base_path, file_name)
    # Metni karakter bazlı parçalara ayır
    text_chunksinChar = convert_Page_ChunkinChar(pdf_texts)
    # Karakter bazlı parçaları token bazlı parçalara ayır
    text_chunksinTokens = convert_Chunk_Token(text_chunksinChar,sentence_transformer_model)
    # Metadata ve ID'leri hazırla
    ids,metadatas = add_meta_data(text_chunksinTokens,file_name,category, current_id)
    # Sonraki belge için başlangıç ID'sini güncelle
    current_id = current_id + len(text_chunksinTokens)
    # Belgeleri ChromaDB koleksiyonuna ekle
    chroma_collection = add_document_to_collection(ids, metadatas, text_chunksinTokens, chroma_collection)
    # Belgenin eklendiğine dair mesaj ve güncel parça sayısını yazdır
    print(f"📖{file_name} koleksiyona eklendi. Kolleksiyondaki mevcut parça sayısı: {chroma_collection.count()}\n")

  print(f"📚Koleksiyona belgeler yüklendi👍\n")
  print(f"📚Kolleksiyondaki mevcut parça sayısı: {chroma_collection.count()}\n")
  # ChromaDB istemcisi ve koleksiyonunu döndür
  return  chroma_client, chroma_collection


#%% iş akışı
collection_name = "Yonergeler"
     

chroma_client, chroma_collection= load_multiple_pdfs_to_ChromaDB(collection_name,sentence_transformer_model)
#%% sorgu
sample_queries=[
    "Muafiyet hakkından vazgeçmek isteyen bir öğrenci nasıl bir başvuru yapmalıdır?",
    "Muafiyetten feragat eden öğrenci tekrar başvuru yapabilir mi?",
    "Lisans ders grupları ve kontenjanlar kim tarafından belirlenir?",
    "Muafiyet başvurusu nasıl yapılır?", # NOTE: Chunk'larda feragat süreci anlatılıyor, başvuru süreci değil.
    "Öğrencinin muafiyet hakkı kaç kere kullanılabilir?" # NOTE: Tekrar başvuru yasağı var ama kullanım sayısı hakkında bilgi yok.
]

query_no=2
print(f"Sorgu: {sample_queries[query_no]}")
chroma_collection.query(
    query_texts=sample_queries[query_no],
    n_results=5,  # Specify the number of results (top k)
)
#%% bağlam ve cevap
import textwrap # Metinleri belirli bir genişliğe göre sarmak için kullanılan kütüphane
from IPython.display import display # Jupyter/Colab ortamında çıktıları daha düzenli göstermek için
from IPython.display import Markdown # Markdown formatında çıktı oluşturmak için

# Normal metni Markdown formatına dönüştüren ve girinti ekleyen fonksiyon
def to_markdown(text):
  # Madde işaretlerini (•) Markdown listesi formatına ( * ) dönüştür
  text = text.replace('•', '  *')
  # Metne her satırın başına '> ' ekleyerek girinti uygula
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# ChromaDB koleksiyonundan ilgili belgeleri sorgulamak için fonksiyon
def retrieveDocs(chroma_collection, query, n_results=5, return_only_docs=False):
    # ChromaDB koleksiyonunu sorgula
    # query_texts: Sorgu metinlerinin listesi
    # include: Döndürülecek bilgileri belirtir (belgeler, metadatalar, mesafeler)
    # n_results: Döndürülecek sonuç sayısı (en benzer ilk k belge)
    results = chroma_collection.query(query_texts=[query],
                                      include= [ "documents","metadatas",'distances' ],
                                      n_results=n_results)

    # Eğer sadece belge içeriği isteniyorsa, ilk belge listesini döndür
    if return_only_docs:
        return results['documents'][0]
    # Aksi takdirde, tüm sonuçları (belgeler, metadatalar, mesafeler) içeren sözlüğü döndür
    else:
        return results

def show_results(results, return_only_docs=False):
  output_string = "" # Çıktıyı biriktirmek için boş bir dize değişkeni oluştur

  # Eğer sadece belge içeriği döndürülmüşse (return_only_docs=True)
  if return_only_docs:
    retrieved_documents = results # Sonuçlar doğrudan belge listesidir
    if len(retrieved_documents) == 0: # Boş liste kontrolü
      output_string += "🚩İlgili Sonuç bulunamadı...\n" # Sonuç bulunamadı mesajını dizeye ekle
      return output_string # Dizeyi döndür
    for i, doc in enumerate(retrieved_documents): # Belgeler üzerinde döngü
      output_string += f"Metin Parçası No: {i+1}: " # Belge numarasını dizeye ekle
      output_string += to_markdown(doc).data + "\n" # Belge metnini Markdown formatında alıp dizeye ekle
  # Eğer tam sonuç sözlüğü döndürülmüşse (return_only_docs=False)
  else:
      retrieved_documents = results['documents'][0] # Belge metinlerini al
      if len(retrieved_documents) == 0: # Boş belge listesi kontrolü
          output_string += "🚩İlgili Sonuç bulunamadı...\n" # Sonuç bulunamadı mesajını dizeye ekle
          return output_string # Dizeyi döndür
      retrieved_documents_metadata = results['metadatas'][0] # Metadata bilgisini al
      retrieved_documents_distances = results['distances'][0] # Mesafe bilgisini al


      for i, doc in enumerate(retrieved_documents): # Belgeler üzerinde döngü
          output_string += f"Metin Parçası No: {i+1}: " # Belge numarasını dizeye ekle
          output_string += to_markdown(doc).data + "\n" # Belge metnini Markdown formatında alıp dizeye ekle
          output_string += f"Metin Parçası Kaynağı: {retrieved_documents_metadata[i]['document']}\n" # Belge kaynağını dizeye ekle
          output_string += f"Metin Parçası Kaynak Tipi: {retrieved_documents_metadata[i]['category']}\n" # Belge kategorisini dizeye ekle
          output_string += f"Metin Parçasının Anlamsal Uzaklığı: {retrieved_documents_distances[i]}\n" # Belge mesafesini dizeye ekle

  return output_string # Oluşturulan dizeyi döndür

query_no=2 # Kullanılacak örnek sorgunun listedeki indeksi
print(f"Sorgu: {sample_queries[query_no]}") # Seçilen sorguyu yazdır

# retrieveDocs fonksiyonunu çağırarak ilgili belgeleri al
# chroma_collection: Sorgu yapılacak ChromaDB koleksiyonu
query= sample_queries[query_no]
top_k=3 # En benzer ilk k belgeyi getir
return_only_docs=False #Sadece belge içeriğini değil, tam sonuç sözlüğünü döndür (metadata ve mesafeler dahil)
retrieved_documents=retrieveDocs(chroma_collection,
                                 query,
                                 n_results=top_k,
                                 return_only_docs=return_only_docs)

# show_results fonksiyonunu çağırarak alınan belgeleri göster
chunks=show_results(retrieved_documents, return_only_docs=return_only_docs)
print(chunks)

#%% sistem promptu

system_prompt = system_prompt = """
Sen Yıldız Teknik Üniversitesi (YTÜ) öğrencilerine yardımcı olan bir akademik asistansın.
Görevin, öğrencilerin akademik ve idari sorularını YALNIZCA sana verilen "Erişilen Belgeler"
bölümündeki metin parçalarına dayanarak cevaplamaktır. Bu metin parçaları YTÜ'nün resmi
yönetmelik ve yönergelerinden alınmıştır.

KURALLAR:

1. Sadece verilen belgeleri kullan.
   - Genel bilgini, tahminlerini veya başka üniversitelerin kurallarını kullanma.
   - Cevap belgelerde açıkça yoksa sadece şunu yaz:
     "Bu sorunun yanıtı elimdeki belgelere göre belirlenemiyor. Lütfen ilgili fakülte veya Öğrenci İşleri Daire Başkanlığı ile iletişime geçin."
   - Asla rakam, süre, tarih veya kural uydurma.

2. Kaynağı belirt.
   - Her metin parçasının altında "Metin Parçası Kaynağı" satırında hangi belgeden geldiği yazar.
   - Cevabının sonunda, bilgiyi hangi belgeden aldığını belge adıyla belirt.

3. Farklı belgelere dikkat et.
   - Metin parçaları farklı yönerge veya yönetmeliklerden gelebilir.
   - Farklı belgeler aynı konuda farklı kurallar içeriyorsa, her kuralın hangi belgeye ait olduğunu ayrı ayrı belirt.
   - Soruyla ilgisiz metin parçalarını görmezden gel.

4. Cevap biçimi.
   - Önce soruya doğrudan ve kısa bir cevap ver.
   - Gerekirse ardından belgelerdeki ilgili ifadeye dayanan kısa bir açıklama ekle.
   - Süre, sayı ve koşulları belgede yazdığı gibi aktar.

5. Metin kusurları.
   - Belgeler PDF'ten çıkarıldığı için kelimelerin içinde hatalı boşluklar olabilir (örneğin "öğrencileri n" = "öğrencilerin").
   - Bu kusurları anlamı bozmadan düzelterek oku, ama cevabında hatalı yazımı tekrarlama.

6. Dil ve üslup.
   - Her zaman Türkçe cevap ver.
   - Resmi, nazik ve anlaşılır bir dil kullan. Öğrenciye "siz" diye hitap et.
"""


user_prompt = f"""
### Kullanıcı Sorgusu:
{query}

### Erişilen Belgeler:
{chunks}
"""

print(user_prompt)

#%% cevap oluşturma

import ollama

def generate_answer(system_prompt, user_message, model_name="gemma3:4b"):
    # Her soru bağımsız: liste sadece talimat + kullanıcı mesajından oluşur
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    # Mesaj listesini yerel Ollama sunucusundaki modele gönder
    response = ollama.chat(model=model_name, messages=messages)
    # Cevabın sadece metin kısmını döndür
    return response["message"]["content"]

llm_answer = generate_answer(system_prompt, user_prompt)
print("\n🤖 Cevap:\n")
print(llm_answer)
# %%
