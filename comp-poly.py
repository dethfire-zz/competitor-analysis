# import modules
import pandas as pd
import streamlit as st
import requests
import re
import json
import base64
from polyfuzz import PolyFuzz
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

st.markdown("""
<style>
.big-font {
    font-size:50px !important;
    label: 20px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<p class="big-font">Competitive URL Analysis</p>
<p>Match your URLs to your competitors, find title keyword and ranking keyword count differences</p>
<b>Directions: </b></ br><ol>
<li>Upload SF crawl CSV of own site</li>
<li>Upload SF crawl CSV of competitor</li>
<li>(optional) Find Semrush API key. API cost = 10 credits * keyword per URL. Keyword reports capped to max 50 keywords per URL. </li>
<li>Not recommended for sites with several thousands pages or more</li>
</ol>
""", unsafe_allow_html=True)

with st.form("data"):
    get_your_crawl = st.file_uploader("Upload your SG CSV",type=['csv'])
    url_branding1 = st.text_input('Enter your title branding','ie. | ABC Tools')
    get_comp_crawl = st.file_uploader("Upload your competitor SG CSV",type=['csv'])
    url_branding2 = st.text_input('Enter your competitor title branding','ie - XYZ Alloys')
    apikey = st.text_input('Enter your semrush api key','')
    submitted = st.form_submit_button("Process")
    
    if submitted:
        # import crawl data
        df_you = pd.read_csv(get_your_crawl)[['Address','Status Code','Indexability Status','Title 1']]
        df_comp = pd.read_csv(get_comp_crawl)[['Address','Status Code','Indexability Status','Title 1']]

        #get first row of each address
        domainrecord1 = df_you["Address"].loc[0]
        domainrecord2 = df_comp["Address"].loc[0]

        #get room domain paths and names
        getdomain1 = re.search('^[^\/]+:\/\/[^\/]*?\.?([^\/.]+)\.[^\/.]+(?::\d+)?\/', domainrecord1 ,re.IGNORECASE)

        if getdomain1:
            domain1 = getdomain1.group(0)
            name1 = getdomain1.group(1)

        getdomain2 = re.search('^[^\/]+:\/\/[^\/]*?\.?([^\/.]+)\.[^\/.]+(?::\d+)?\/', domainrecord2 ,re.IGNORECASE)

        if getdomain2:
            domain2 = getdomain2.group(0)
            name2 = getdomain2.group(1)

        df_you = df_you[~df_you['Address'].str.contains('page')]
        df_you = df_you[df_you['Status Code'] == 200]
        df_you = df_you[df_you['Indexability Status'] != 'nondex']
        df_you['Address'] = df_you['Address'].str.replace(domain1,'')
        df_you['Title 1'] = df_you['Title 1'].str.replace(url_branding1,'')

        df_comp = df_comp[~df_comp['Address'].str.contains('page')]
        df_comp = df_comp[df_comp['Status Code'] == 200]
        df_comp = df_comp[df_comp['Indexability Status'] != 'noindex']
        df_comp['Address'] = df_comp['Address'].str.replace(domain2,'')
        df_comp['Title 1'] = df_comp['Title 1'].str.replace(url_branding2,'')

        #convert address df to lists
        comp_list = df_comp['Address'].tolist()
        you_list = df_you['Address'].tolist()
        
        st.write("(1/4) Importing and cleaning data... complete")


        # match urls
        model = PolyFuzz("EditDistance")
        model.match(you_list, comp_list)

        df_results = model.get_matches()
        df_results = df_results.sort_values(by='Similarity', ascending=False)
        df_results["Similarity"] = df_results["Similarity"].round(3)

        index_names = df_results[ df_results['Similarity'] < .7 ].index
        df_results.drop(index_names, inplace = True)
        
        st.write("(2/4) Fuzzy URL Matching... complete")


        # merge dfs
        df_merge1 = pd.merge(df_results, df_you, left_on='From', right_on='Address', how='inner')
        df_merge1 = df_merge1.drop(columns=['Address', 'Status Code'])
        df_merge1 = df_merge1.rename(columns={"From": name1 +" URL", "To": name2 + " URL", "Title 1": name1 +" Title"})

        df_merge2 = pd.merge(df_merge1, df_comp, left_on= name2 + ' URL', right_on='Address', how='inner')
        df_merge2 = df_merge2.drop(columns=['Address', 'Status Code'])
        df_merge2 = df_merge2.rename(columns={"Title 1": name2 + " Title"})

        # clean up df
        df_merge2 = df_merge2[[name1 + " URL"] + [name1 + " Title"] + [name2 + " URL"] + [name2 + " Title"] + ["Similarity"]]
        df_merge2[name1 + " URL"] = domain1 + df_merge2[name1 + " URL"]
        df_merge2[name2 + " URL"] = domain2 + df_merge2[name2 + " URL"]

        # tokenize titles 1
        title_token = df_merge2[name1 + " Title"].tolist()

        for z in range(len(title_token)):
          title_token[z] = title_token[z].split(" ")

          try:
            title_token[z].remove('')
          except:
            pass

          title_token[z] = nltk.pos_tag(title_token[z])
          
          title_token[z] = [x for x in title_token[z] if x[1] in ['NN','NNS','NNP','NNPS','VB','VBD']]
          title_token[z] = [x for x in title_token[z] if x[0] != '|']
          title_token[z] = [x[0] for x in title_token[z]]
          
        #tokenize titles 2
        title_token2 = df_merge2[name2 + " Title"].tolist()

        for z in range(len(title_token2)):
          title_token2[z] = title_token2[z].split(" ")

          try:
            title_token2[z].remove('')
          except:
            pass

          title_token2[z] = nltk.pos_tag(title_token2[z])
          
          title_token2[z] = [x for x in title_token2[z] if x[1] in ['NN','NNS','NNP','NNPS','VB','VBD']]
          title_token2[z] = [x for x in title_token2[z] if x[0] != '|']
          title_token2[z] = [x[0] for x in title_token2[z]]

        #title difference
        title_diff = []
        for x in range(len(title_token)):
          diff = list(set(title_token[x]) - set(title_token2[x])) + list(set(title_token2[x]) - set(title_token[x]))
          title_diff.append(diff)

        diff_series = pd.Series(title_diff)

        df_merge2["Title Difference"] = diff_series
        
        st.write("(3/4) Finding Title Difference... complete")


        #urls to list
        df_comp3 = df_merge2[name2 + " URL"].tolist()
        df_you3 = df_merge2[name1 + " URL"].tolist()

        # get num of keywords per url 1
        keyword_count = []

        for x in df_comp3:
          url = f'https://api.semrush.com/?type=url_organic&key={apikey}&display_limit=50&url={x}&database=us'

          payload={}
          headers = {}

          response = requests.request("GET", url, headers=headers, data=payload)
          response = response.text.split("\n")
          keyword_count.append(len(response)-2)

        df_merge2[name2 + " Keywords"] = keyword_count

        # get num of keywords per url 2
        keyword_count2 = []

        for x in df_you3:
          url = f'https://api.semrush.com/?type=url_organic&key={apikey}&display_limit=50&url={x}&database=us'

          payload={}
          headers = {}

          response = requests.request("GET", url, headers=headers, data=payload)
          response = response.text.split("\n")
          keyword_count2.append(len(response)-2)

        df_merge2[name1 + " Keywords"] = keyword_count2

        #add keyword diff to df
        keydiff = [m - n for m,n in zip(keyword_count2,keyword_count)]

        df_merge2["Keyword Difference"] = keydiff
        
        if apikey !="":
            st.write("(4/4) Finding URL Keyword Difference... complete")
        else:
            st.write("(4/4) Finding URL Keyword Difference... :warning: No API Key")

        def get_csv_download_link(df, title):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            return f'<a href="data:file/csv;base64,{b64}" download="{title}">Download CSV File</a>'
    

        #save and display
        df_merge2 = df_merge2[[name1 + " URL"] + [name1 + " Title"] + [name1 + " Keywords"] + [name2 + " URL"] + [name2 + " Title"] + [name2 + " Keywords"] + ["Keyword Difference"] + ["Title Difference"] + ["Similarity"]]
        st.markdown(get_csv_download_link(df_merge2,name1 + "VS" + name2 +".csv"), unsafe_allow_html=True)

        st.dataframe(df_merge2)
        
st.write('Author: [Greg Bernhardt](https://twitter.com/GregBernhardt4) | Friends: [Rocket Clicks](https://www.rocketclicks.com) and [Physics Forums](https://www.physicsforums.com)')
