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



continues=['C2_Acresowned','A10_boys','A15_income','D3_LH_income','D15_Nearest_Market_km',\
'F3_Distance_Water_km','F3_Distance_Water_min','B3_FCS','B13_MAHFP','B2_HDDS']
specific=['D17']


def main():	
	
	topic = st.sidebar.selectbox('What do you want to do ?',['Nothing','Display correlations','Display Wordclouds','Display Sankey Graphs','Design my own visuals'])
	
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
		
		col1.image(wc.to_array(),width=600,heigth=300)	
		
		if col1.checkbox('Would you like to filter Wordcloud according to other questions'):
		
			feature2=st.selectbox('Select one question to filter the wordcloud (Select on of the last ones for checking some new tools)',[questions[i] for i in data.columns if \
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
				subcol1.image(wc1.to_array(),width=600,heigth=300)
				subcol2.write('Response over the threshold')
				subcol2.image(wc2.to_array(),width=600,heigth=300)
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
						subcol1.image(wc2.to_array(),width=600,heigth=300)
					else:
						subcol2.write('Response : '+str(L[i])+' '+str(len(data[data[var2]==L[i]]))+' '+'repondent')
						subcol2.image(wc2.to_array(),width=600,heigth=300)
			
			
			
	else:
		st.title('\t WARDI \t July 2021')	
		


if __name__== '__main__':
    main()
    
