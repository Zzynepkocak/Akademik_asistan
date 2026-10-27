# YTÜ Akademik Asistan (RAG Chatbot)

Yıldız Teknik Üniversitesi'nin yönetmelik ve yönergelerinden (PDF) beslenen, Türkçe soru-cevap yapan bir RAG (Retrieval-Augmented Generation) asistanı. Tamamen yerelde çalışır; API anahtarı gerekmez.

## Nasıl çalışıyor?

```
PDF → Metin çıkarma → Chunking (karakter + token) → Metadata → Embedding → ChromaDB → Retrieval → LLM (Ollama) → Cevap
```

| Aşama | Kullanılan araç |
|---|---|
| PDF'ten metin | PyPDF2 |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (1500 karakter) + `SentenceTransformersTokenTextSplitter` (128 token) |
| Embedding | `distiluse-base-multilingual-cased-v1` (sentence-transformers) |
| Vektör veritabanı | ChromaDB (kalıcı, diskte) |
| Cevap üretimi | Ollama + `gemma3:4b` |

Her chunk, geldiği belgenin adını metadata olarak taşır; model cevabında kaynağı belirtir.

## Kurulum

1. Depoyu klonlayın ve sanal ortam oluşturun:
   ```bash
   git clone <repo-adresi>
   cd akademik-asistan
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```
2. [Ollama](https://ollama.com)'yı kurun ve modeli indirin:
   ```bash
   ollama pull gemma3:4b
   ```
3. YTÜ yönerge PDF'lerini YTÜ'nün resmi web sitesinden indirip `data/` klasörüne koyun. (PDF'ler depoya dahil edilmemiştir.)

## Çalıştırma

`src/rag_pipeline.py` dosyası `# %%` hücrelerinden oluşur; VS Code'un interaktif penceresinde hücre hücre ya da baştan sona çalıştırılabilir.

- İlk çalıştırmada PDF'ler okunur, chunk'lanır ve `vector_db/` klasörüne yazılır.
- Soru, dosyadaki `sample_queries` listesinden `query_no` ile seçilir.

## Bilinen eksikler / yol haritası

- [ ] Chunk metadata'sına sayfa numarası eklemek
- [ ] Veritabanını her çalıştırmada yeniden oluşturmak yerine var olanı açmak
- [ ] PDF metin kusurlarını (kelime içi boşluklar, sayfa başlıkları) temizlemek
- [ ] Kullanıcıdan terminal üzerinden soru almak
- [ ] Retrieval kalitesini iyileştirmek (ör. `top_k`, uzaklık eşiği)
