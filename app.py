import json
from typing import List, Dict, Any
import pandas as pd
import streamlit as st

from product_research import analyze_urls
from validators import ProductResearch

st.set_page_config(page_title="URL → Product Research", layout="wide")
st.title("🔎 Product Research (URL-first)")

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("LLM Provider", ["OpenAI", "Gemini"], index=0)
    openai_model = st.text_input("OpenAI model", value="gpt-4o-mini")
    gemini_model = st.text_input("Gemini model", value="gemini-1.5-pro")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    st.caption("Set environment variables before running:")
    st.code("OPENAI_API_KEY=...\nGOOGLE_API_KEY=...")

st.markdown("Paste **one or more** product page URLs (one per line):")
urls_text = st.text_area(
    "Product URLs",
    height=140,
    placeholder="https://example.com/product/abc\nhttps://brand.com/item/xyz"
)

col1, col2, col3 = st.columns([1,1,1])
with col1:
    run_btn = st.button("Run Analysis", type="primary")
with col2:
    demo_btn = st.button("Try Demo URL")
with col3:
    clear_btn = st.button("Clear")

if demo_btn:
    urls_text = "https://www.apple.com/airpods-pro/"
if clear_btn:
    urls_text = ""
    st.experimental_rerun()

def normalize_urls(blob: str) -> List[str]:
    out = []
    for line in (blob or "").splitlines():
        u = line.strip()
        if u.startswith("http://") or u.startswith("https://"):
            out.append(u)
    # dedupe, keep order
    return list(dict.fromkeys(out))

if run_btn:
    urls = normalize_urls(urls_text)
    if not urls:
        st.warning("Please paste at least one valid URL.")
    else:
        with st.spinner("Fetching pages and extracting specs..."):
            results, pages = analyze_urls(
                urls=urls,
                provider=provider,
                model=gemini_model if provider == "Gemini" else openai_model,
                temperature=temperature
            )

        for i, (page, product) in enumerate(zip(pages, results), start=1):
            with st.expander(f"Result {i}: {page.url}", expanded=True):
                st.subheader("Page snapshot")
                st.write(f"**Title:** {page.title or '—'}")
                st.caption(f"HTTP: {page.status_code} · Extracted chars: {len(page.text or ''):,}")
                st.text_area("Extracted Text (preview)", (page.text or "")[:4000], height=200)
                st.subheader("Structured Output")
                st.json(json.loads(product.model_dump_json()))

        if results:
            st.success(f"Parsed {len(results)} page(s).")

            st.markdown("### 🧮 Comparison")
            rows: List[Dict[str, Any]] = []
            for r in results:
                rows.append({
                    "Name": r.name or "",
                    "Brand": r.brand or "",
                    "Category": r.category or "",
                    "Price": r.pricing or "",
                    "URL": r.source_url
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            # Downloads
            import io, csv
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # CSV
            csv_buf = io.StringIO()
            writer = csv.writer(csv_buf)
            writer.writerow([
                "name","brand","category","key_specifications",
                "unique_selling_points","target_use_cases",
                "forbidden_or_unsupported_claims","pricing","source_url"
            ])
            for it in results:
                writer.writerow([
                    it.name or "",
                    it.brand or "",
                    it.category or "",
                    " | ".join(it.key_specifications or []),
                    " | ".join(it.unique_selling_points or []),
                    " | ".join(it.target_use_cases or []),
                    " | ".join(it.forbidden_or_unsupported_claims or []),
                    it.pricing or "",
                    it.source_url
                ])
            st.download_button(
                "Download CSV",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name=f"product_research_{ts}.csv",
                mime="text/csv"
            )

            # JSON
            st.download_button(
                "Download JSON",
                data=json.dumps([json.loads(it.model_dump_json()) for it in results], ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"product_research_{ts}.json",
                mime="application/json"
            )
