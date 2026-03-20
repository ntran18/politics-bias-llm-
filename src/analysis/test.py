import streamlit as st
import pandas as pd
import os
from datetime import datetime
from utils import load_and_clean_data, GRAPHS_DIR

st.set_page_config(layout="wide", page_title="LLM Bias Reviewer")

def main():
    st.title("🔍 High-Conflict Article Deep-Dive")
    
    # 1. Load Data
    @st.cache_data
    def get_data():
        return load_and_clean_data()
    
    df = get_data()

    # 2. Stats Calculation
    article_stats = df.groupby('index').agg({
        'llm_label_bin': 'mean',
        'human_label_bin': 'first',
        'source': 'first'
    })
    article_stats['conflict_score'] = 0.5 - abs(article_stats['llm_label_bin'] - 0.5)

    # Sidebar Filters
    st.sidebar.header("Filter Articles")
    min_conflict = st.sidebar.slider("Min Conflict Score", 0.0, 0.5, 0.25)
    controversial_indices = article_stats[article_stats['conflict_score'] >= min_conflict].index.tolist()
    
    selected_idx = st.sidebar.selectbox("Select Article Index", controversial_indices)

    # 3. Article Metadata Section
    # Pulling info from the first row of that index
    # article_infos = pd.read_csv(os.path.join("./data/", "clean_original_data.csv"))
    article_info = df[df['index'] == selected_idx].iloc[0]
    
    st.header(f"Article: {article_info.get('url', 'URL not found')}")
    st.caption(f"Index: {selected_idx} | Source: {article_info.get('source', 'Unknown')}")
    
    with st.expander("📄 View Original Article Text"):
        st.write(article_info.get('text', 'Article text not found in dataframe.'))

    # 4. Metrics
    m1, m2, m3 = st.columns(3)
    truth = article_stats.loc[selected_idx, 'human_label_bin']
    ai_mean = article_stats.loc[selected_idx, 'llm_label_bin']
    
    m1.metric("Human Ground Truth", "Biased" if truth == 1 else "Not Biased")
    m2.metric("AI Bias Consensus", f"{ai_mean*100:.0f}%")
    m3.metric("Conflict Score", round(article_stats.loc[selected_idx, 'conflict_score'], 3))

    st.divider()

    # 5. Side-by-Side Model Explanations
    st.subheader("Model-Specific Reasoning (+all Context)")
    
    # Filter for +all level and one stable version
    comparison_df = df[(df['index'] == selected_idx) & 
                       (df['detail_level'] == '+all') & 
                       (df['version'].str.contains('v1|7.1'))].sort_values('llm_model')

    model_cols = st.columns(len(comparison_df))
    
    for i, (idx, row) in enumerate(comparison_df.iterrows()):
        with model_cols[i]:
            color = "red" if row['llm_label_bin'] == 1 else "blue"
            st.markdown(f"#### :{color}[{row['llm_model']}]")
            st.markdown(f"**Verdict:** {'Biased' if row['llm_label_bin'] == 1 else 'Not Biased'}")
            st.markdown(f"**Confidence:** {row['llm_confidence']}%")
            st.info(row['llm_explanation'])

    # 6. Researcher Coding Note
    st.divider()
    st.subheader("📝 Qualitative Coding Notes")
    
    note = st.text_area("What is the primary driver of disagreement here?", 
                        help="Analyze if they focus on 'source name' vs 'adjectives' vs 'framing'.")
    
    if st.button("💾 Save Observation"):
        note_entry = pd.DataFrame([{
            'timestamp': datetime.now(),
            'article_index': selected_idx,
            'human_truth': truth,
            'ai_mean': ai_mean,
            'observation': note
        }])
        
        notes_file = os.path.join(GRAPHS_DIR, "manual_coding_notes.csv")
        
        # Append to existing or create new
        if os.path.exists(notes_file):
            note_entry.to_csv(notes_file, mode='a', header=False, index=False)
        else:
            note_entry.to_csv(notes_file, index=False)
            
        st.success(f"Note saved for article {selected_idx}!")

if __name__ == "__main__":
    main()