import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
import pickle
import pydeck as pdk
import re
from collections import Counter
from PIL import Image

#import variables

#########################  a faire #########################################
# 
#
###########################################################################"


st.set_page_config(layout="wide")


#import des données
@st.cache
def load_data():
	data = pd.read_csv('viz.csv',sep='\t')
	correl=pd.read_csv('correlations_validated.csv',sep='\t')
	questions=pd.read_csv('questions.csv',sep='\t').iloc[0].to_dict()
	
	return data,correl,questions

data,correl,questions=load_data()

#st.dataframe(correl)
#st.write(data.columns)
#st.write(correl.shape)

def sankey_graph(data,L,height=600,width=1600):
    """ sankey graph de data pour les catégories dans L dans l'ordre et 
    de hauter et longueur définie éventuellement"""
    
    nodes_colors=["blue","green","grey",'yellow',"coral"]
    link_colors=["lightblue","lightgreen","lightgrey","lightyellow","lightcoral"]
    
    
    labels=[]
    source=[]
    target=[]
    
    for cat in L:
        lab=data[cat].unique().tolist()
        lab.sort()
        labels+=lab
    
    for i in range(len(data[L[0]].unique())): #j'itère sur mes premieres sources
    
        source+=[i for k in range(len(data[L[1]].unique()))] #j'envois sur ma catégorie 2
        index=len(data[L[0]].unique())
        target+=[k for k in range(index,len(data[L[1]].unique())+index)]
        
        for n in range(1,len(L)-1):
        
            source+=[index+k for k in range(len(data[L[n]].unique())) for j in range(len(data[L[n+1]].unique()))]
            index+=len(data[L[n]].unique())
            target+=[index+k for j in range(len(data[L[n]].unique())) for k in range(len(data[L[n+1]].unique()))]
       
    iteration=int(len(source)/len(data[L[0]].unique()))
    value_prov=[(int(i//iteration),source[i],target[i]) for i in range(len(source))]
    
    
    value=[]
    k=0
    position=[]
    for i in L:
        k+=len(data[i].unique())
        position.append(k)
    
   
    
    for triplet in value_prov:    
        k=0
        while triplet[1]>=position[k]:
            k+=1
        
        df=data[data[L[0]]==labels[triplet[0]]].copy()
        df=df[df[L[k]]==labels[triplet[1]]]
        #Je sélectionne ma première catégorie
        value.append(len(df[df[L[k+1]]==labels[triplet[2]]]))
        
    color_nodes=nodes_colors[:len(data[L[0]].unique())]+["black" for i in range(len(labels)-len(data[L[0]].unique()))]
    print(color_nodes)
    color_links=[]
    for i in range(len(data[L[0]].unique())):
    	color_links+=[link_colors[i] for couleur in range(iteration)]
    print(L,len(L),iteration)
    print(color_links)
   
   
    fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 30,
      line = dict(color = "black", width = 1),
      label = [i.upper() for i in labels],
      color=color_nodes
      )
      
    ,
    link = dict(
      source = source, # indices correspond to labels, eg A1, A2, A1, B1, ...
      target = target,
      value = value,
      color = color_links))])
    return fig



continues=['C2_Acresowned','A10_boys','A15_income','D3_LH_income','D15_Nearest_Market_km',\
'F3_Distance_Water_km','F3_Distance_Water_min','B3_FCS','B13_MAHFP','B2_HDDS']
specific=['D17']

img1 = Image.open("logoAxiom.jpeg")
img2 = Image.open("logoWardi.png")

def main():	
	
	st.sidebar.image(img1,width=200)
	topic = st.sidebar.radio('What do you want to do ?',('Display correlations','Display Wordclouds','Display Sankey Graphs','Design my own visuals'))
	
	title1, title2 = st.beta_columns([5,1])
	title2.image(img2)
	
	
	if topic=='Display correlations':
	
		
		for var in [i for i in correl['Variables'].unique()]:
		
			
		
			if var=='Village_clean':
		
				st.title('Correlations related to the village of the respondent')
					
				st.write('FCS scores is very different from one village to another')
		
				df=data[['Village_clean','longitude','latitude','FCS Score']]
				a=df.groupby(['Village_clean','FCS Score']).aggregate({'longitude':'count'}).unstack()
				a.columns=['Acceptable','Borderline','Poor']
				b=df.groupby(['Village_clean']).aggregate({'longitude':'mean','latitude':'min'})
				df=a.merge(b,how='left',left_index=True,right_index=True).fillna(0)
				df['village']=df.index
				
				st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/light-v9',\
				initial_view_state=pdk.ViewState(latitude=4.55,longitude=45.35,zoom=10,pitch=50,),\
				layers=[pdk.Layer('ColumnLayer',data=df,get_position=['longitude','latitude'],get_elevation='Acceptable+Borderline+Poor',\
				elevation_scale=400,pickable=True,auto_highlight=True,get_fill_color=[255, 0, 0],radius=300),\
				pdk.Layer('ColumnLayer',data=df,get_position=['longitude','latitude'],get_elevation='Borderline+Poor',\
				elevation_scale=400,pickable=True,auto_highlight=True,get_fill_color=[255,255 , 0],radius=300),\
				pdk.Layer('ColumnLayer',data=df,get_position=['longitude','latitude'],get_elevation='Poor',\
				elevation_scale=400,pickable=True,auto_highlight=True,get_fill_color=[0, 255, 0],radius=300),\
				pdk.Layer("TextLayer",data=df,get_position=['longitude','latitude'],filled=False,billboard=False,get_line_color=[180, 180, 180],\
				get_text="village",get_label_size=200,get_label_color=[0,0,0],line_width_min_pixels=1,)
				],\
				))
				st.write('PS: Position of villages is not the real one as I did not have the GPS coordinates for the benficiaries. This was just to show what could be done')
				col1, col2, col3 = st.beta_columns([4,1,4])
				x=df['village']
				fig = go.Figure(go.Bar(x=x, y=df['Poor'], name='Poor',marker_color='red'))
				fig.add_trace(go.Bar(x=x, y=df['Borderline'], name='Borderline',marker_color='yellow'))
				fig.add_trace(go.Bar(x=x, y=df['Acceptable'], name='Acceptable',marker_color='green'))
				fig.update_layout(barmode='relative', \
	        	          xaxis={'title':'Village'},\
        		          yaxis={'title':'Persons'}, legend_title_text='FCS Score')
				col1.plotly_chart(fig)
				somme=df['Poor']+df['Borderline']+df['Acceptable']
				fig2 = go.Figure(go.Bar(x=x, y=df['Poor']/somme, name='Poor',marker_color='red'))
				fig2.add_trace(go.Bar(x=x, y=df['Borderline']/somme, name='Borderline',marker_color='yellow'))
				fig2.add_trace(go.Bar(x=x, y=df['Acceptable']/somme, name='Acceptable',marker_color='green'))
				fig2.update_layout(barmode='relative', \
	        	          xaxis={'title':'Village'},\
        		          yaxis={'title':'Pourcentage'}, legend_title_text='FCS Score')
				col3.plotly_chart(fig2)
		
		
		
		
			else:
				st.title('Correlation with question '+questions[var])
			
				for correlation in correl[correl['Variables']==var]['Correl'].unique():
				
					if correlation in specific:
						st.write(correl[(correl['Variables']==var) & (correl['Correl']==correlation)]['Description'].iloc[0])
						
						df=pd.DataFrame(columns=['Challenge','Acres Owned','Chall'])
						dico={'D17_challenge_roads':'Roads','D17_challenge_Insecurity':'Insecurity',\
      						'D17_challenge_transpCost':'Transportation Costs','D17_challenge_distance':'Distance',\
      						'D17_challenge_notenough':'Not enough goods', 'D17_challenge_Other':'Other'}
						for i in dico:
							a=data[[i,'C2_Acresowned']].copy()
							a=a[a[i].isin(['Yes','No'])]
							
							a.columns=['Challenge','Acres Owned']
							a['Chall']=a['Challenge'].apply(lambda x: dico[i])
							df=df.append(a)
						#st.write(df)
						fig = px.box(df, x="Chall", y="Acres Owned",color='Challenge')
						fig.update_layout(barmode='relative',xaxis={'title':'Challenges for accessing market'},\
										yaxis_title='Acres owned',width=1000,height=600)
						st.plotly_chart(fig)
						
					else:
						st.write(correl[(correl['Variables']==var) & (correl['Correl']==correlation)]['Description'].iloc[0])
						df=data[[correlation,var]].copy()
					
						if var in continues:
							if correlation in continues:
								fig = px.scatter(df, x=var, y=correlation)
								fig.update_layout(xaxis={\
								'title':questions[var]},yaxis_title=questions[correlation],width=1500,height=800)
							else:
								fig = px.box(df, x=correlation, y=var,points='all')
								fig.update_traces(marker_color='green')
								fig.update_layout(barmode='relative', \
        	        	  				xaxis={'title':questions[correlation]},\
        	        	 				yaxis_title=questions[var],width=1000,height=600)
				
							st.plotly_chart(fig)
				
						else:
							if correlation in continues:
								fig = px.box(df, x=var, y=correlation,points='all')
								fig.update_traces(marker_color='green')
								fig.update_layout(barmode='relative',xaxis={'title':questions[var]},\
										yaxis_title=questions[correlation],width=1000,height=600)
								st.plotly_chart(fig)
                 				
							else:
								agg=df[[correlation,var]].groupby(by=[var,correlation]).aggregate({var:'count'}).unstack()
								x=[i for i in agg.index]
								fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green'))
								for i in range(len(agg.columns)-1):
    									fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1]))
								fig.update_layout(barmode='relative', \
        	        	  				xaxis={'title':questions[var]},\
        	        	  				yaxis={'title':'Persons'}, legend_title_text=None)
								
								agg=df[[correlation,var]].groupby(by=[var,correlation]).aggregate({var:'count'}).unstack()
								agg=agg.T/agg.T.sum()
								agg=agg.T*100
								x=[i for i in agg.index]
								fig2 = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green'))
								for i in range(len(agg.columns)-1):
    									fig2.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1]))
								fig2.update_layout(barmode='relative', \
        	        	  				xaxis={'title':questions[var]},\
        	        	  				yaxis={'title':'Pourcentages'}, legend_title_text=None)
						
								st.plotly_chart(fig)
								st.plotly_chart(fig2)
						
						
						
	elif topic=='Display Wordclouds':
		#st.write('coucou')
		df=data[[i for i in data.columns if 'text' in i]].copy()
		#st.write(df)
		feature=st.sidebar.selectbox('Select the question for which you would like to visualize wordclouds of answers',[questions[i] for i in df.columns])	
		var=[i for i in questions if questions[i]==feature][0]
		
		col1, col3 = st.beta_columns([6,3])
		col1.title('Wordcloud from question:')
		col1.title(feature)
				
		x, y = np.ogrid[:300, :300]
		mask = ((x - 150)) ** 2 + ((y - 150)/1.4) ** 2 > 130 ** 2
		mask = 255 * mask.astype(int)
		corpus=' '.join(df[var].apply(lambda x:'' if x=='0' else x))
		corpus=re.sub('[^A-Za-z ]',' ', corpus)
		corpus=re.sub('\s+',' ', corpus)
		corpus=corpus.lower()
		
		col3.title('')
		col3.title('')
		col3.title('')
		sw=col3.multiselect('Select words you would like to remove from the wordcloud', [i[0] for i in Counter(corpus.split(' ')).most_common()[:20] if i[0] not in STOPWORDS])
		
		if corpus==' ':
    			corpus='No_response'
		else:
			corpus=' '.join([i for i in corpus.split(' ') if i not in sw])
		
		wc = WordCloud(background_color="white", repeat=False, mask=mask)
		
		wc.generate(corpus)
		
		col1.image(wc.to_array(),width=400,heigth=200)	
		
		if col1.checkbox('Would you like to filter Wordcloud according to other questions'):
		
			feature2=st.selectbox('Select one question to filter the wordcloud (Select one of the last ones for checking some new tools)',[questions[i] for i in data.columns if \
			i!='FCS Score' and (i in continues or len(data[i].unique())<=8)])
			var2=[i for i in questions if questions[i]==feature2][0]
			
			if var2 in continues:
				threshold=st.slider('Select the threshold', min_value=data[var2].fillna(0).min(),max_value=data[var2].fillna(0).max())
				subcol1,subcol2=st.beta_columns([2,2])	
				
				corpus1=' '.join(data[data[var2]<threshold][var].apply(lambda x:'' if x=='0' else x))
				corpus1=re.sub('[^A-Za-z ]',' ', corpus1)
				corpus1=re.sub('\s+',' ', corpus1)
				corpus1=corpus1.lower()
				if corpus1==' 'or corpus1=='':
    					corpus1='No_response'
				else:
					corpus1=' '.join([i for i in corpus.split(' ') if i not in sw])
				wc1 = WordCloud(background_color="white", repeat=False, mask=mask)
				wc1.generate(corpus1)
				corpus2=' '.join(data[data[var2]>=threshold][var].apply(lambda x:'' if x=='0' else x))
				corpus2=re.sub('[^A-Za-z ]',' ', corpus2)
				corpus2=re.sub('\s+',' ', corpus2)
				corpus2=corpus2.lower()
				if corpus2==' ' or corpus2=='':
    					corpus2='No_response'
				else:
					corpus2=' '.join([i for i in corpus.split(' ') if i not in sw])
				wc2 = WordCloud(background_color="white", repeat=False, mask=mask)
				wc2.generate(corpus2)
				subcol1.write('Response under the threshold')
				subcol1.image(wc1.to_array(),width=400,heigth=200)
				subcol2.write('Response over the threshold')
				subcol2.image(wc2.to_array(),width=400,heigth=200)
			else:
				subcol1,subcol2=st.beta_columns([2,2])
				L=data[var2].unique()
				
				corpus1=corpus2=corpus3=corpus4=corpus5=corpus6=corpus7=corpus8=''
				Corpuses=[corpus1,corpus2,corpus3,corpus4,corpus5,corpus6,corpus7,corpus8]
				
				
				for i in range(len(L)):
					Corpuses[i]=' '.join(data[data[var2]==L[i]][var].apply(lambda x:'' if x=='0' else x))
					Corpuses[i]=re.sub('[^A-Za-z ]',' ', Corpuses[i])
					Corpuses[i]=re.sub('\s+',' ', Corpuses[i])
					Corpuses[i]=Corpuses[i].lower()
					if Corpuses[i]==' ':
    						Corpuses[i]='No_response'
					else:
						Corpuses[i]=' '.join([i for i in Corpuses[i].split(' ') if i not in sw])
					wc2 = WordCloud(background_color="white", repeat=False, mask=mask)
					wc2.generate(Corpuses[i])
					if i%2==0:
						subcol1.write('Response : '+str(L[i])+' '+str(len(data[data[var2]==L[i]]))+' '+'repondent')
						subcol1.image(wc2.to_array(),width=400,heigth=200)
					else:
						subcol2.write('Response : '+str(L[i])+' '+str(len(data[data[var2]==L[i]]))+' '+'repondent')
						subcol2.image(wc2.to_array(),width=400,heigth=200)
			
	elif topic=='Display Sankey Graphs':
	
		st.title('Visuals for questions related to cultures (questions C3 to C17)')
		st.title('')
				
			
		sankey=[i for i in data.columns if i[0]=='C' and 'C1_' not in i and 'C2_' not in i and i!='Clan']
		sankeyseeds=sankey[:65]
		sank=data[sankeyseeds]
		bean=sank[[i for i in sank.columns if 'Bean' in i]].copy()
		sesame=sank[[i for i in sank.columns if 'Sesame' in i]].copy()
		cowpea=sank[[i for i in sank.columns if 'Cowpea' in i]].copy()
		maize=sank[[i for i in sank.columns if 'Maize' in i]].copy()
		other=sank[[i for i in sank.columns if 'Other' in i]].copy()
		colonnes=['Seeds Planted','Type of seeds','Origin of seeds','Area cultivated','Did you have enough seed',\
          'Did you face pest attack','Area affected','Have you done pest management','Origin of fertilizer',\
          'Fertilizer from Wardi','Applied good practices','Used irrigation','Area irrigated']
		for i in [bean,sesame,cowpea,maize,other]:
    			i.columns=colonnes
		bean=bean[bean['Seeds Planted']=='Yes']
		sesame=sesame[sesame['Seeds Planted']=='Yes']
		cowpea=cowpea[cowpea['Seeds Planted']=='Yes']
		maize=maize[maize['Seeds Planted']=='Yes']
		other=other[other['Seeds Planted']=='Yes']
		
		bean['Seeds Planted']=bean['Seeds Planted'].apply(lambda x: 'Beans')
		sesame['Seeds Planted']=sesame['Seeds Planted'].apply(lambda x: 'Sesame')
		cowpea['Seeds Planted']=cowpea['Seeds Planted'].apply(lambda x: 'Cowpeas')
		maize['Seeds Planted']=maize['Seeds Planted'].apply(lambda x: 'Maize')
		other['Seeds Planted']=other['Seeds Planted'].apply(lambda x: 'Other')
		
		sank=pd.DataFrame(columns=colonnes)
		for i in [bean,sesame,cowpea,maize,other]:
		    sank=sank.append(i)
		sank['ones']=np.ones(len(sank))
		
		
		
		
		st.title('Some examples')
		
		st.write('Seeds planted - Origin of Seeds - Type of Seeds - Area Cultivated - Did you have enough seeds?')
		fig=sankey_graph(sank,['Seeds Planted','Origin of seeds','Type of seeds','Area cultivated','Did you have enough seed'],height=600,width=1500)
		fig.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
		
		st.plotly_chart(fig)
		
		st.write('Origin of fertilizer - Did you face pest attack - Applied good practices - Seeds Planted')
		fig1=sankey_graph(sank,['Origin of fertilizer','Did you face pest attack','Applied good practices','Seeds Planted'],height=600,width=1500)
		fig1.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
		
		st.plotly_chart(fig1)
		
		st.write('Area Cultivated - Type of Seeds - Did you face pest attack - Area affected')
		fig2=sankey_graph(sank,['Area cultivated','Type of seeds','Did you face pest attack','Area affected'],height=600,width=1500)
		fig2.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
		
		st.plotly_chart(fig2)
		
		if st.checkbox('Design my own Sankey Graph'):
			
			feats=st.multiselect('Select features you want to see in the order you want them to appear', colonnes)
			
			if len(feats)>=2:
				st.write(' - '.join(feats))
				fig3=sankey_graph(sank,feats,height=600,width=1500)
				fig3.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
				st.plotly_chart(fig3)
		
		
		
			
	
	
	else:
		st.title('\t WARDI \t July 2021')	


    
 
if __name__== '__main__':
    main()




    
