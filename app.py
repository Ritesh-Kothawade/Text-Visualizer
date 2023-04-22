# core packages
# import en_core_web_sm
import streamlit as st
import streamlit.components.v1 as stc
st.set_option('deprecation.showfileUploaderEncoding',False)
st.set_option('deprecation.showPyplotGlobalUse', False)
from ui_template import HTML_BANNER,HTML_BANNER_SKEWED,HTML_WRAPPER,HTML_STICKER

# EDA packages

import pandas as pd
import neattext as nt
import neattext.functions as nfx
from collections import Counter

# Data viz packages
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# import altair as alt

# NLP packahes
from wordcloud import WordCloud
import nltk
import spacy
# nlp = en_core_web_sm.load()
nlp = spacy.load('en_core_web_sm')
from spacy import displacy

# Text Viz packages
from tagvisualizer import TagVisualizer  
from yellowbrick.text import PosTagVisualizer
# Fxn


def plot_wordcloud(docx):
    mywordcloud = WordCloud().generate(docx)
    plt.imshow(mywordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot()


def plot_mendelhall_curve(docx):
    # word Length Distribution
    word_length = [len(token) for token in docx.split()]
    word_length_counts = Counter(word_length)
    sorted_word_length_count = sorted(dict(word_length_counts).items())
    x, y = zip(*sorted_word_length_count)
    fig = plt.figure(figsize=(20, 10))
    plt.plot(x, y)
    plt.title("Plot of Word Length Distribution(Mendelhall Curve)")
    plt.show()
    st.pyplot(fig)


def get_most_common_tokens(docx, num=10):
    word_freq = Counter(docx.split())
    most_common_tokens = word_freq.most_common(num)
    return dict(most_common_tokens)


def generate_tags(docx):
    tagged_docx = [[nltk.pos_tag(nltk.word_tokenize(i))] for i in docx.split('.')]
    return tagged_docx


def plot_most_common_tokens(docx, num=10):
    word_freq = Counter(docx.split())
    most_common_tokens = word_freq.most_common(num)
    x, y = zip(*most_common_tokens)
    fig = plt.figure(figsize=(20, 10))
    plt.bar(x, y)
    plt.title("Plot of Most Common Tokens")
    plt.show()
    st.pyplot(fig)


def plot_pos_tags(tagged_docx):
    # Create Visualizer, Fit, Score, Show
    pos_visualizer = PosTagVisualizer()
    pos_visualizer.fit(tagged_docx)
    pos_visualizer.show()
    st.pyplot()


def main():
    """Text Visualization NLP app"""

    st.title("Text Visualizer NLP App")
    menu = ["Home", "DropFiles", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Home':
        st.subheader("Home")
        raw_text = st.text_area("Enter Text Here")
        viz_task = ["Basic", "WordCloud",
                    "Mendelhall Curve", "Pos Tagger", "NER"]
        viz_choice = st.sidebar.selectbox("Choice", viz_task)
        if st.button("Visualize"):
            if viz_choice == "WordCloud":
                plot_wordcloud(raw_text)

            elif viz_choice == "Pos Tagger":
                tagged_docx = generate_tags(raw_text)
                # st.write(tagged_docx)
                plot_pos_tags(tagged_docx)
                t = TagVisualizer(raw_text)
                stc.html(t.visualize_tags())

            elif viz_choice == "Mendelhall Curve":
                plot_mendelhall_curve(raw_text)

            elif viz_choice == "NER":
                docx = nlp(raw_text)
                html = displacy.render(docx, style="ent")
                html = html.replace("\n\n", "\n")
                result = HTML_WRAPPER.format(html)
                stc.html(result)

            else:
                st.info("Text Visualizer")
                processed_text = nfx.remove_stopwords(raw_text)
                word_desc = nt.TextFrame(raw_text).word_stats()
                st.info("Text Description")
                st.write(word_desc)

                st.info("Most Common Tokens")
                most_common_tokens = get_most_common_tokens(processed_text)
                token_df = pd.DataFrame(
                    most_common_tokens.items(), columns=['Tokens', 'Counts'])
                st.dataframe(token_df)
                plot_most_common_tokens(processed_text)

    elif choice == "DropFiles":
        st.subheader("Drag and Drop Files")
        raw_text_file = st.file_uploader("Upload Text Files", type=['txt'])
        if st.button("Visualize"):
            if raw_text_file is not None:
                viz_task = ["Basic", "Wordcloud",
                            "Mendelhall Curve", "Pos Tagger", "NER"]
                viz_choice = st.sidebar.selectbox("Choice", viz_task)
                raw_text = raw_text_file.read()
                st.write(raw_text)
                if viz_choice == "WordCloud":
                    plot_wordcloud(raw_text)

                elif viz_choice == "Pos Tagger":
                    tagged_docx = generate_tags(raw_text)
                    # st.write(tagged_docx)
                    plot_pos_tags(tagged_docx)
                    t = TagVisualizer(raw_text)
                    stc.html(t.visualize_tags())

                elif viz_choice == "Mendelhall Curve":
                    plot_mendelhall_curve(raw_text)

                elif viz_choice == "NER":
                    docx = nlp(raw_text)
                    html = displacy.render(docx, style="ent")
                    html = html.replace("\n\n", "\n")
                    result = HTML_WRAPPER.format(html)
                    stc.html(result)

                else:
                    st.info("Text Visualizer")
                    processed_text = nfx.remove_stopwords(raw_text)
                    word_desc = nt.TextFrame(raw_text).word_stats()
                    st.info("Text Description")
                    st.write(word_desc)

                    st.info("Most Common Tokens")
                    most_common_tokens = get_most_common_tokens(processed_text)
                    token_df = pd.DataFrame(
                        most_common_tokens.items(), columns=['Tokens', 'Counts'])
                    st.dataframe(token_df)
                    plot_most_common_tokens(processed_text)
    else:
        st.subheader("About App")
        stc.html(HTML_STICKER, width=800, height=600)


if __name__ == '__main__':
    main()
