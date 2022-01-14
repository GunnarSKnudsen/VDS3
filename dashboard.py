import dash
from dash import dcc
from dash import html
#import dash_core_components as dcc
#import dash_html_components as html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output# Load Data

import json

import numpy as np

import dash
import dash_table as dt
from dash import html
from dash import dcc
import plotly.graph_objects as go
import pandas as pd


from sklearn.metrics import mean_absolute_percentage_error
import plotly.figure_factory as ff


# figure out the index thing
#df = pd.read_csv('dataframe_to_visualize.csv', index_col=0, parse_dates=True)
df = pd.read_csv('dataframe_to_visualize_subset.csv', index_col=0, parse_dates=True)
#df = pd.read_csv('dataframe_to_visualize.csv', index_col=0, parse_dates=True)
print(f"Read in dataframe with shape {df.shape}")

# Add final data processing


## Add a dummy key, so plotly can reference:
df['dummyKey'] = range(len(df))

## Recode Gender
df['Sex'] = df['Sex'].map({'F': 'Female', 'M': 'Male'})

## Gather Year (Accidentally deleted it in previous notebook)
df['CompetitionYear'] = pd.DatetimeIndex(df['Date']).year

## Round for datatable
df['predicted'] = df['predicted'].round(1)
df['Mean height'] = df['Mean height'].round(1)

## Codefy nulls to unknown
df['Continent'] = df['Continent'].fillna('Unknown')
df['Country'] = df['Country'].fillna('Unknown')
df['Event'] = df['Event'].fillna('Unknown')
df['Equipment'] = df['Equipment'].fillna('Unknown')
df['ParentFederation'] = df['ParentFederation'].fillna('Unknown')
df['Sex'] = df['Sex'].fillna('Unknown')
print(df)


# Initialize the app
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True
app.title = "186.868 Visual Data Science - Dashboard"

# Function for generating dropdown values
def get_options(vals):
	dict_list = []
	for i in vals:
		dict_list.append({'label': str(i), 'value': str(i)})
	return(dict_list)

# Helping myself debugging
print(df.columns)


# Initialize dropdowns to all values, before callback is called first time.
## Enables selective filtering
continent_options_all = get_options(df['Continent'].unique())
country_options_all = get_options(df['Country'].unique())
event_options_all = get_options(df['Event'].unique())
equipment_options_all = get_options(df['Equipment'].unique())
parent_federation_options_all = get_options(df['ParentFederation'].unique())
valid_res_options_all = get_options(df['Valid_Results'].unique())
sex_options_all = get_options(df['Sex'].unique())

continent_options = continent_options_all
country_options = country_options_all
event_options = event_options_all
equipment_options = equipment_options_all
parent_federation_options = parent_federation_options_all
valid_res_options = valid_res_options_all
sex_options = sex_options_all 

print(valid_res_options)

print("Generated dropdown options")


tableColumns = ['Name', 'Country', 'Sex', 'Age', 'BodyweightKg', 'Date', 'Best3SquatKg','Best3BenchKg', 'Best3DeadliftKg', 'TotalKg', 'Wilks','Mean height', 'predicted']
tableColumns = ['Name', 'Country', 'Sex', 'Age', 'BodyweightKg', 'predicted', 'TotalKg', 'Mean height']

pd.options.display.float_format = '${:.2f}'.format
#df['predicted']=tableColumns['predicted'].map('{:,.2f}%'.format)

#Index(['Name', 'Continent', 'Country', 'Sex', 'Age', 'BodyweightKg',
#       'Valid_Results', 'Date', 'Event', 'Equipment', 'Best3SquatKg',
#       'Best3BenchKg', 'Best3DeadliftKg', 'TotalKg', 'Dots', 'Wilks',
#       'Glossbrenner', 'Goodlift', 'Mean height', 'ParentFederation',
#       'predicted', 'dummyKey', 'CompetitionYear'],
#      dtype='object')



date_min = df['Date'].min()
date_max = df['Date'].max()
print(date_min, date_max)

bodyweight_min = df['BodyweightKg'].min()
bodyweight_max = df['BodyweightKg'].max()
#print(bodyweight_min, bodyweight_max)
#bodyweight_list = df['BodyweightKg'].unique()

height_min = df['Mean height'].min()
height_max = df['Mean height'].max()

print("Generated slider options")

drilldownfilter_options = get_options(['No', 'Yes'])
axis_options_kde = get_options([
'BodyweightKg', 'Mean height', 'TotalKg', 'Dots', 'Wilks', 'Age', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'Glossbrenner', 'Goodlift', 'predicted', 'CompetitionYear'])
grouping_options_kde = get_options(['Sex', 'Continent', 'ParentFederation', 'Equipment', 'Country', 'Event'])





app.layout = html.Div(
    children=[
        html.Div(className='row',
                 children=[
                 html.H1("Powerlifting Results, Total, Weight, (and height)", style={'background-color': '#333333'}),

                    html.Div(className='four columns div-user-controls bg-grey', style={'margin': '5px', 'padding':'5px'},
                             children=[
                                 #html.H2('Parameters', style={'color': '#00ff00'}),
                                 html.Div(
                                     className='div-for-dropdown',
                                     children=[
                                     	########################
                                     	## Drop down selectors #
                                     	########################
                                     	html.Div(children=[
	                                     	html.Div(children=[
	                                     		html.H3("Max Rows:", className='h3-for-filter'),
				                             	dcc.Input(id="nSamplesTotal",
												type="number",
												value=1000
												, style={'backgroundColor': '#1E1E1E', 'width':'98%', 'color':'#FFFFFF'}
												),
				                             ], style={'width':'32%', 'display':'inline-block', 'vertical-align': 'top'}), 
				                             html.Div(children=[
				                             	html.H3("DD Filters:", className='h3-for-filter'),
		                                         dcc.Dropdown(id='drillDownFilterSelector'
		                                         			, options=drilldownfilter_options
		                                         			, multi=False
		                                                    , value=drilldownfilter_options[0].get('value')
		                                                    , style={'backgroundColor': '#1E1E1E'}
		                                                    , className='dropDownSelector'
		                                                    , clearable=False
		                                                      ),
				                             	], style={'width':'32%', 'display':'inline-block', 'vertical-align': 'top'}), 
				                             html.Div(children=[
				                             	html.H3("Valid Results:", className='h3-for-filter'),
		                                         dcc.Dropdown(id='validResSelector'
		                                         			, options=valid_res_options
		                                         			, multi=True
		                                         			, value='True'
		                                         			, style={'backgroundColor': '#1E1E1E'}
		                                         			, className='dropDownSelector'
		                                                      ),
				                             	], style={'width':'32%', 'display':'inline-block', 'vertical-align': 'top'}), 
				                             html.Div(id='rowFoundDiv', className='h3-for-filter'),
			                             ]),

                                     	

                                     	html.Div(children=[
	                                     	html.Div(
	                                     		children=[
	                                     			html.H3("Continent:", className='h3-for-filter'),
			                                         dcc.Dropdown(id='continentSelector'
			                                         			, options=continent_options
			                                         			, multi=True
			                                                    #, value=continent_options[0].get('value')
			                                                    , style={'backgroundColor': '#1E1E1E'}
			                                                    , className='dropDownSelector'
			                                                      ),
				                             	], style={'width':'49%', 'display':'inline-block', 'vertical-align': 'top'}
				                			), 
				                			html.Div(children=[
						                       		html.H3("Country:", className='h3-for-filter'),
						                       		dcc.Dropdown(id='countrySelector'
                                         			, options=country_options
                                         			, multi=True
                                         			#, value=country_options[0].get('value')
                                         			, style={'backgroundColor': '#1E1E1E'}
                                         			, className='dropDownSelector'
                                                      ),
				                             	], style={'width':'49%', 'display':'inline-block', 'vertical-align': 'top'}), 
			                             ]),

                                     	html.Div(children=[
	                                     	html.Div(
	                                     		children=[
	                                     			html.H3("Event:", className='h3-for-filter'),
			                                         dcc.Dropdown(id='eventSelector'
			                                         			, options=event_options
			                                         			, multi=True
			                                         			, value='SBD'
			                                         			, style={'backgroundColor': '#1E1E1E'}
			                                         			, className='dropDownSelector'
			                                                      ),
				                             	], style={'width':'49%', 'display':'inline-block', 'vertical-align': 'top'}
				                			), 
				                			html.Div(children=[
						                       		html.H3("Equipment:", className='h3-for-filter'),
			                                         dcc.Dropdown(id='equipmentSelector'
			                                         			, options=equipment_options
			                                         			, multi=True
			                                         			#, value=equipment_options[0].get('value')
			                                         			, style={'backgroundColor': '#1E1E1E'}
			                                         			, className='dropDownSelector'
			                                                      ),
				                             	], style={'width':'49%', 'display':'inline-block', 'vertical-align': 'top'}), 
			                             ]),

                                     	html.Div(children=[
	                                     	html.Div(
	                                     		children=[
	                                     			html.H3("Federation:", className='h3-for-filter'),
			                                         dcc.Dropdown(id='parentFederationSelector'
			                                         			, options=parent_federation_options
			                                         			, multi=True
			                                         			#, value=parent_federation_options[0].get('value')
			                                         			, style={'backgroundColor': '#1E1E1E'}
			                                         			, className='dropDownSelector'
			                                                      ),
				                             	], style={'width':'49%', 'display':'inline-block', 'vertical-align': 'top'}
				                			), 
				                			html.Div(children=[
						                       		html.H3("Sex:", className='h3-for-filter'),
			                                         dcc.Dropdown(id='sexSelector'
			                                         			, options=sex_options
			                                         			, multi=True
			                                         			#, value=sex_options[0].get('value')
			                                         			, style={'backgroundColor': '#1E1E1E'}
			                                         			, className='dropDownSelector'
			                                                      ),
				                             	], style={'width':'49%', 'display':'inline-block', 'vertical-align': 'top'}), 
			                             ]),


                                     	html.Div(children=[
	                                     	html.Div(
	                                     		children=[
	                                     			html.H3("Bodyweight:", className='h3-for-filter'),
			                                         dcc.RangeSlider(
																	id="slider2",
																	min=bodyweight_min,
																	max=bodyweight_max,
																	step=1,
																	#marks={i: str(i) for i in bodyweight_list},
																	included=True,
																	allowCross=False,
																	#marks=[str(bodyweight_min), str(bodyweight_max)],
																	value=[bodyweight_min, bodyweight_max],
																	),
			                                         #html.Div(id='WeightSliderRange'),
				                             	], style={'width':'49%', 'display':'inline-block', 'vertical-align': 'top'}
				                			), 
				                			html.Div(children=[
						                       		html.H3("Mean Height:", className='h3-for-filter'),
			                                         dcc.RangeSlider(
																	id="slider3",
																	min=height_min,
																	max=height_max,
																	step=1,
																	#marks={i: str(i) for i in bodyweight_list},
																	included=True,
																	allowCross=False,
																	#marks=[str(bodyweight_min), str(bodyweight_max)],
																	value=[height_min, height_max],
																	),
			                                         #html.Div(id='HeightSliderRange')
				                             	], style={'width':'49%', 'display':'inline-block', 'vertical-align': 'top'}), 
			                             ]),

                                         
                                         
                                         #######################################################
                                         #######################################################
                                         html.Div(children=[
	                                     	html.Div(
	                                     		children=[
	                                     			html.H3("X-Axis:", className='h3-for-filter'),
													dcc.Dropdown(id='xAxisSelectorKDE'
					                                   			, options=axis_options_kde
					                                         	, multi=False
					                                         	, value='BodyweightKg' #'Mean height'
					                                         	, style={'backgroundColor': '#1E1E1E'}
					                                         	, className='dropDownSelector'
					                                         	, clearable=False
					                                         	),
				                             	], style={'width':'32%', 'display':'inline-block', 'vertical-align': 'top'}
				                			), 
				                			html.Div(children=[
						                       		html.H3("Y-Axis:", className='h3-for-filter'),
													dcc.Dropdown(id='yAxisSelectorKDE'
					                                   			, options=axis_options_kde
					                                         	, multi=False
					                                         	, value='TotalKg'
					                                         	, style={'backgroundColor': '#1E1E1E'}
					                                         	, className='dropDownSelector'
					                                         	, clearable=False
					                                         	),
				                             	], style={'width':'32%', 'display':'inline-block', 'vertical-align': 'top'}
				                             ), 
				                			html.Div(children=[
						                       		html.H3("Grouping:", className='h3-for-filter'),
													dcc.Dropdown(id='colorGroupingKDE'
					                                   			, options=grouping_options_kde
					                                         	, multi=False
					                                         	, value='Sex'
					                                         	, style={'backgroundColor': '#1E1E1E'}
					                                         	, className='dropDownSelector'
					                                         	, clearable=False
					                                         	),
				                             	], style={'width':'32%', 'display':'inline-block', 'vertical-align': 'top'}
				                             ), 
			                             ]
			                            )
								, dcc.Graph(id='fig3Div', config={'displayModeBar': True}, animate=True)
                                         
                                         #######################################################
                                         #html.H3("Year:", className='h3-for-filter'),
                                         #html.H3("Age:", className='h3-for-filter'),
                                         #html.H3("Total:", className='h3-for-filter'),
                                     ],
                                     style={'color': '#1E1E1E'})
                                ]
                             ),
                    html.Div(className='four columns div-for-charts bg-grey',
                             children=[
                                 dcc.Graph(id='timeseries', config={'displayModeBar': True}, animate=True),
                                 html.Div(children=[
                                 			#html.H2("Hovered:"),
								            #html.Pre(id='hover-data'),
											dt.DataTable(
												id='hover-data', data=df.head(0).to_dict('records'),
											columns=[{"name": i, "id": i} for i in tableColumns],
											style_cell_conditional=[
										        {
										            'if': {'column_id': c},
										            'textAlign': 'left'
										        } for c in ['Date', 'Region']
										    ],
										    style_data={
										        'color': 'black',
										        'backgroundColor': 'white',
										        'height': 'auto',
										        'width': '100%'
										    },
										    style_cell={
										        'overflow': 'hidden',
										        'textOverflow': 'ellipsis',
										        'maxWidth': '100%'
										    },
										    style_data_conditional=[
										        {
										            'if': {'row_index': 'odd'},
										            'backgroundColor': 'rgb(220, 220, 220)',
										        }
										    ],
										    style_header={
										        'backgroundColor': 'rgb(210, 210, 210)',
										        'color': 'black',
										        'fontWeight': 'bold'
										    }
											),
								        ], className='six columns'
								        , style={'width':'32%', 'padding':'0px', 'margin':'0px', 'padding-top':'5px', 'padding-bottom':'5px'}
								        ),
                                 dcc.Graph(id='timeseries2', config={'displayModeBar': True}, animate=True),

                             ]
                             ),
                    html.Div(className='three columns div-for-charts2 bg-grey',
                    	children=[
                    			dcc.Graph(id='modelPredictionsGraph', config={'displayModeBar': True}, animate=True),
                    			html.Div(id='mapeDiv')
								, html.Div(children=[html.H2("Selected:"),
								            #html.Pre(id='selected-data'),
								            dt.DataTable(
												id='selected-data', data=df.head(0).to_dict('records'),
											columns=[{"name": i, "id": i} for i in tableColumns],
											style_cell_conditional=[
										        {
										            'if': {'column_id': c},
										            'textAlign': 'left'
										        } for c in ['Date', 'Region']
										    ],
										    style_data={
										        'color': 'black',
										        'backgroundColor': 'white',
										    },
										    style_data_conditional=[
										        {
										            'if': {'row_index': 'odd'},
										            'backgroundColor': 'rgb(220, 220, 220)',
										        }
										    ],
										    style_table={'overflowX': 'auto'},
										    style_header={
										        'backgroundColor': 'rgb(210, 210, 210)',
										        'color': 'black',
										        'fontWeight': 'bold'
										    }
											),
								        ]),
                             ]
                             )
                    ]),

        ]

)

# Hover callback:
# When hovering on the scatterplot, current point is described in table below.
@app.callback(
    Output("hover-data", "data"),
    Input('timeseries', 'hoverData'))
def display_hover_data(hoverData):
	# Only act if point is hovered
	if hoverData is not None:
		# Find which points are hovered by looping through all (1)
		indices = []
		for p in hoverData['points']:
			custIndex = p['customdata'][0]
			indices.append(custIndex)

		df_selected = df[df.dummyKey.isin(indices)]

		return df_selected.to_dict('records')


# Select callback:
# When selecting points on the scatterplot, (brushing), I'm linking to predictions plot
## Also showing table of selected data
@app.callback(
	[Output("selected-data", "data"),
	 Output('modelPredictionsGraph', 'figure'),
	 Output('mapeDiv', 'children')],
	Input('timeseries', 'selectedData'))
def display_selected_data(selectedData):
	#Only return if anything is selected
	if selectedData is not None:
		# Get keys of all selected points
		indices = []
		for p in selectedData['points']:
			custIndex = p['customdata'][0]
			indices.append(custIndex)

		# Filter for selected points
		df_selected = df[df.dummyKey.isin(indices)]

		print(df_selected.shape)

		# Calculate MAPE for selected points
		mape = mean_absolute_percentage_error(df_selected.BodyweightKg, df_selected.predicted)
		mapeString = "MAPE= {:.0%}".format(mape)

		df_corr = df_selected.corr()
		print(df_corr)

		mask = np.triu(np.ones_like(df_corr, dtype=bool))
		df_mask = df_corr.mask(mask)

		figure = ff.create_annotated_heatmap(z=df_mask.to_numpy(), 
                                  x=df_mask.columns.tolist(),
                                  y=df_mask.columns.tolist(),
                                  colorscale=px.colors.diverging.RdBu,
                                  hoverinfo="none", #Shows hoverinfo for null values
                                  showscale=True, ygap=1, xgap=1
                                 )

		figure.update_xaxes(side="bottom")

		figure.update_layout(
		    title_text='Heatmap', 
		    title_x=0.5, 
		    width=1000, 
		    height=1000,
		    xaxis_showgrid=False,
		    yaxis_showgrid=False,
		    xaxis_zeroline=False,
		    yaxis_zeroline=False,
		    yaxis_autorange='reversed',
		    template='plotly_white'
		)




		# (Re)generate the plot
		#figure = px.scatter(
		#	df_selected
		#	, x="BodyweightKg"
		#	, y=["predicted"]
		#	, color="Sex"
		#	, custom_data = ['dummyKey']
		#	, trendline="ols"
		#	, title="Model - Weight vs predicted"
		#)

		figure = go.Heatmap(
			z=df_corr.values,#to_numpy(),
			x=df_corr.index.values,#columns.tolist(),
			y=df_corr.columns.values)#index.tolist(),
			#zmax=1, zmin=-1,
			#showscale=True,
			#hoverongaps=True
		#)

		fig = ff.create_annotated_heatmap(z=df_corr.to_numpy(), 
										x=df_corr.columns.tolist(),
										y=df_corr.columns.tolist(),
										colorscale=px.colors.diverging.RdBu,
										hoverinfo="none", #Shows hoverinfo for null values
										showscale=True, ygap=1, xgap=1
										)

		fig.update_xaxes(side="bottom")

		fig.update_layout(
			title_text='Heatmap', 
			title_x=0.5, 
			#width=1000, 
			#height=1000,
			xaxis_showgrid=False,
			yaxis_showgrid=False,
			xaxis_zeroline=False,
			yaxis_zeroline=False,
			yaxis_autorange='reversed',
			template='plotly_white'
		)

# NaN values are not handled automatically and are displayed in the figure
# So we need to get rid of the text manually
		for i in range(len(fig.layout.annotations)):
			if fig.layout.annotations[i].text == 'nan':
				fig.layout.annotations[i].text = ""


		fig = px.scatter_3d(df_selected, x='BodyweightKg', y='Mean height', z='TotalKg',
				color='Sex', symbol='Sex')


		# Update layout
		fig.update_layout({
			'plot_bgcolor': '#161616',
			'paper_bgcolor': '#161616',
			'template':'plotly_dark'
		})

		#draw a square
		x = [0, 1, 0, 1, 0, 1, 0, 1]
		y = [0, 1, 1, 0, 0, 1, 1, 0]
		z = [0, 0, 0, 0, 1, 1, 1, 1]

		x = df_selected['Mean height'].values
		y1 = df_selected['BodyweightKg'].values
		y2 = df_selected['predicted'].values
		z = df_selected['TotalKg'].values
		print(y1)

		#the start and end point for each line
		pairs = [(0,6), (1,7)]

		trace1 = go.Scatter3d(
			x=x,
			y=y1,
			z=z,
			mode='markers',
			marker=dict(color="blue"),
			name='BW (True)'#, 			color="red"
		)
		

		x_lines = list()
		y_lines = list()
		z_lines = list()

		for i in range(len(x)):
			x_lines.append(x[i])
			x_lines.append(x[i])
			y_lines.append(y1[i])
			y_lines.append(y2[i])
			z_lines.append(z[i])
			z_lines.append(z[i])
			x_lines.append(None)
			y_lines.append(None)
			z_lines.append(None)

		trace2 = go.Scatter3d(
			x=x,
			y=y2,
			z=z,
			mode='markers',
			marker=dict(color="red"),
			name='BW (Pred)'#, 			color="red"
		)

		trace3 = go.Scatter3d(
			x=x_lines,
			y=y_lines,
			z=z_lines,
			mode='lines',
			name=""
		)

		fig = go.Figure(data=[trace1, trace2, trace3])

		fig.update_layout(legend=dict(
		orientation="h",
		yanchor="bottom",
		y=1.02,
		xanchor="right",
		x=1
		))

		fig.update_layout(scene=dict(xaxis=dict(title_text='Mean height (cm)'),
									yaxis=dict(title_text='Bodywieght (kg)'),
									zaxis=dict(title_text='Total (kg)'),
			))

	

		print("I did create something")

				# Update layout
		fig.update_layout({
			'plot_bgcolor': '#161616',
			'paper_bgcolor': '#161616',
			'template':'plotly_dark'
		})

		#https://stackoverflow.com/questions/42301481/adding-specific-lines-to-a-plotly-scatter3d-plot

		# Default option
		#figure.update_layout(
			#dragmode='select',
		#)

		# Ensure a square plot
		#axis_min = min(df_selected.BodyweightKg.min(), df_selected.predicted.min()) * 0.95
		#axis_max = min(df_selected.BodyweightKg.max(), df_selected.predicted.max()) * 1.05
		#figure.update_xaxes(range=[axis_min, axis_max])
		#figure.update_yaxes(range=[axis_min, axis_max])

		# Over and out
		return df_selected.to_dict('records'), fig, mapeString


# "Main" callback
# Update whenever filters are modified
## Outputs three plots, as well as optionally drill down of filters
@app.callback([Output('timeseries', 'figure')
			,  Output('timeseries2', 'figure')
			,  Output('fig3Div', 'figure')
			,  Output('continentSelector', 'options')
			,  Output('countrySelector', 'options')
			,  Output('eventSelector', 'options')
			,  Output('equipmentSelector', 'options')
			,  Output('parentFederationSelector', 'options')
			,  Output('validResSelector', 'options')
			,  Output('sexSelector', 'options')
			,  Output('rowFoundDiv', 'children')
			],[Input('drillDownFilterSelector', 'value')
			,  Input('nSamplesTotal', 'value')
			,  Input('continentSelector', 'value')
			,  Input('countrySelector', 'value')
			,  Input('eventSelector', 'value')
			,  Input('equipmentSelector', 'value')
			,  Input('parentFederationSelector', 'value')
			,  Input('validResSelector', 'value')
			,  Input('sexSelector', 'value')
			,  Input('slider2', 'value')
			,  Input('slider3', 'value')
			#-- Used for KDE plot
			,  Input('xAxisSelectorKDE', 'value')
			,  Input('yAxisSelectorKDE', 'value')
			,  Input('colorGroupingKDE', 'value')
              ])
def update_output(drillDownFilterSelector
				, nSamplesTotal
				, continentSelector
				, countrySelector
				, eventSelector
				, equipmentSelector
				, parentFederationSelector
				, validResSelector
				, sexSelector
				, slider2
				, slider3
				, xAxisSelectorKDE
				, yAxisSelectorKDE
				, colorGroupingKDE
				):
	print("Calling callback")

	# Cleanse parameters:
	## If initialized with single value, it is a string
	## If not set, I want an empty list - makes it easier further down
	if isinstance(continentSelector, str):
		continentSelector = [continentSelector]
	if continentSelector is None:
		continentSelector = []
	#------------------------------------#
	if isinstance(countrySelector, str):
		countrySelector = [countrySelector]
	if countrySelector is None:
		countrySelector = []
	#------------------------------------#
	if isinstance(eventSelector, str):
		eventSelector = [eventSelector]
	if eventSelector is None:
		eventSelector = []
	#------------------------------------#
	if isinstance(equipmentSelector, str):
		equipmentSelector = [equipmentSelector]
	if equipmentSelector is None:
		equipmentSelector = []
	#------------------------------------#
	if isinstance(parentFederationSelector, str):
		parentFederationSelector = [parentFederationSelector]
	if parentFederationSelector is None:
		parentFederationSelector = []
	#------------------------------------#
	if isinstance(validResSelector, str):
		validResSelector = [validResSelector]
	if validResSelector is None:
		validResSelector = []
	#------------------------------------#
	if isinstance(sexSelector, str):
		sexSelector = [sexSelector]
	if sexSelector is None:
		sexSelector = []

	# Update the slider values
	## These are no longer printed, but used to be under the sliders
	WeightSliderRangeValue = str(slider2)
	HeightSliderRangeValue = str(slider3)

	# Clone dataframe for refiltering
	df_filtered = df.copy()	

	# Filter dataset based on inputs
	if len(continentSelector) > 0:
		df_filtered = df_filtered[df_filtered.Continent.isin(continentSelector)]
	if len(countrySelector) > 0:
		df_filtered = df_filtered[df_filtered.Country.isin(countrySelector)]
	if len(eventSelector) > 0:
		df_filtered = df_filtered[df_filtered.Event.isin(eventSelector)]
	if len(equipmentSelector) > 0:
		df_filtered = df_filtered[df_filtered.Equipment.isin(equipmentSelector)]
	if len(parentFederationSelector) > 0:
		df_filtered = df_filtered[df_filtered.ParentFederation.isin(parentFederationSelector)]
	if len(sexSelector) > 0:
		df_filtered = df_filtered[df_filtered.Sex.isin(sexSelector)]
	if len(validResSelector) > 0: # Not proud of this one, but it works.. Should be switched from dropdown
		filtering_list = []
		# Create boolean list from strings
		## Weird hack, as I would otherwise return boolean to the dropdown, which breaks a lot
		for item in validResSelector:
			if item == 'True':
				filtering_list.append(True)
			if item == 'False':
				filtering_list.append(False)
		df_filtered = df_filtered[df_filtered.Valid_Results.isin(filtering_list)]
	
	# Filter based on slider ranges
	df_filtered = df_filtered[df_filtered.BodyweightKg.fillna(0)>= slider2[0]]	
	df_filtered = df_filtered[df_filtered.BodyweightKg.fillna(0)<= slider2[1]]	

	df_filtered = df_filtered[df_filtered['Mean height'].fillna(0)>= slider3[0]]	
	df_filtered = df_filtered[df_filtered['Mean height'].fillna(0)<= slider3[1]]	

	# Count how many rows have been found
	rowsFound = df_filtered.shape[0]

	# Do a subselect after filtering.
	if nSamplesTotal > df_filtered.shape[0]:
		nSamplesTotal = df_filtered.shape[0]
	df_filtered = df_filtered.sample(nSamplesTotal, replace=False)
	rowsShown = df_filtered.shape[0]

	# Refilter filters
	## If drilldown is enabled, we need to find new values to select
	if drillDownFilterSelector == 'No':
		continent_options = continent_options_all
		country_options = country_options_all
		event_options = event_options_all
		equipment_options = equipment_options_all
		parent_federation_options = parent_federation_options_all
		valid_res_options = valid_res_options_all
		sex_options = sex_options_all
	else:
		country_options = get_options(df_filtered['Country'].unique())
		continent_options = get_options(df_filtered['Continent'].unique())
		country_options = get_options(df_filtered['Country'].unique())
		event_options = get_options(df_filtered['Event'].unique())
		equipment_options = get_options(df_filtered['Equipment'].unique())
		parent_federation_options = get_options(df_filtered['ParentFederation'].unique())
		valid_res_options = get_options(df_filtered['Valid_Results'].unique())
		sex_options = get_options(df_filtered['Sex'].unique())

	# Create the plots	

	## Standard scatterplot
	figure = px.scatter(
        df_filtered
        , x="BodyweightKg"
        , y="TotalKg"
        , color="Continent"
        # Append key for filtering
        , custom_data = ['dummyKey']
        , title="Continent"
    )

	## Update layout
	figure.update_layout({
		'plot_bgcolor': '#161616',
		'paper_bgcolor': '#161616',
		'template':'plotly_dark'
	})

    ## Default action
	figure.update_layout(
		dragmode='select',
	)

	# KDE Plot - based on selectors
	figure3 = px.density_contour(
        df_filtered
        , x=xAxisSelectorKDE
        , y=yAxisSelectorKDE
        , color=colorGroupingKDE
        , title="Distribution stuff"
        , marginal_x="histogram"
		, marginal_y="histogram"
    )

	## Update layout
	figure3.update_layout({
		'plot_bgcolor': '#161616',
		'paper_bgcolor': '#161616',
		'template':'plotly_dark'
	})
    ## Default action
	figure3.update_layout(
		dragmode='select',
	)

	# Do aggregation
	df_grouped = df_filtered.groupby(['CompetitionYear'], as_index = False)[['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']].mean()
	
	# Show timeline
	figure2 = px.area(
        df_grouped
        , x="CompetitionYear"
        , y=["Best3SquatKg", "Best3BenchKg", "Best3DeadliftKg"]
        , title="Average score over time"
	)
	
	## Update layout
	figure2.update_layout({
		'plot_bgcolor': '#161616',
		'paper_bgcolor': '#161616',
		'template':'plotly_dark'
		})
	## Default action
	figure2.update_layout(
		dragmode='select',
	)

	# Generate counter text in the filters
	if rowsFound > nSamplesTotal:
		counterText = 'Found ' + str(rowsFound) + ' competitors matching filter. Showing ' + str(rowsShown) + ' competitors'
	else:
		counterText = 'Found ' + str(rowsFound) + ' competitors matching filter.'
	
	# Return everything
	return figure, figure2, figure3\
	, continent_options, country_options, event_options, equipment_options, parent_federation_options, valid_res_options, sex_options\
	, counterText
	#, WeightSliderRangeValue, HeightSliderRangeValue
	
if __name__ == '__main__':
    app.run_server(debug=True)