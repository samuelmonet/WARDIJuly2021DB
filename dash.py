import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
import pydeck as pdk

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
	
		select = st.sidebar.selectbox('Select the specific feature for which you want to see correlations', [questions[i] for i in correl['Variables'].unique()])
		var=[i for i in questions if questions[i]==select][0]
			
		
		if var=='Village_clean':
		
			st.title('Correlation between village and FCS Score')
					
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
			
			
			for correlation in correl[correl['Variables']==var]['Correl'].unique():
				
				if correlation in specific:
					st.title('Correlation between questions '+questions[var] +' and '+'D17 Challenges to acces the market')
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
									yaxis_title='Acres owned',width=1500,height=800)
					st.plotly_chart(fig)
					
									
				else:
					st.title('Correlation between questions '+questions[var] +' and '+questions[correlation])
					st.write(correl[(correl['Variables']==var) & (correl['Correl']==correlation)]['Description'].iloc[0])
					df=data[[correlation,var]].copy()
					
					if var in continues:
						if correlation in continues:
							fig = px.scatter(df, x=var, y=correlation)
							fig.update_layout(xaxis={'title':questions[var]},yaxis_title=questions[correlation],width=1500,height=800)
						else:
							fig = px.box(df, x=correlation, y=var,points='all')
							fig.update_traces(marker_color='green')
							fig.update_layout(barmode='relative', \
        	          				xaxis={'title':questions[correlation]},\
        	         				yaxis_title=questions[var],width=1500,height=800)
				
						st.plotly_chart(fig)
				
					else:
						if correlation in continues:
							fig = px.box(df, x=var, y=correlation,points='all')
							fig.update_traces(marker_color='green')
							fig.update_layout(barmode='relative',xaxis={'title':questions[var]},\
									yaxis_title=questions[correlation],width=1500,height=800)
							st.plotly_chart(fig)
                 				
						else:
							col1, col2, col3 = st.beta_columns([4,1,4])
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
						
							col1.plotly_chart(fig)
							col3.plotly_chart(fig2)
						
						
						
						
						
						
						
						
						
	
	else:
		st.title('\t WARDI \t July 2021')	
		


if __name__== '__main__':
    main()
    
