import dash
from dash import dcc
from dash import html
#import dash_core_components as dcc
#import dash_html_components as html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output# Load Data

import json

import dash
import dash_table as dt
from dash import html
from dash import dcc
import plotly.graph_objects as go
import pandas as pd


from sklearn.metrics import mean_absolute_percentage_error



# figure out the index thing
#df = pd.read_csv('dataframe_to_visualize.csv', index_col=0, parse_dates=True)
df = pd.read_csv('dataframe_to_visualize_subset.csv', index_col=0, parse_dates=True)
df = pd.read_csv('dataframe_to_visualize.csv', index_col=0, parse_dates=True)
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
#Do this https://dash.plotly.com/dash-core-components/rangeslider

# Callback for timeseries price
#@app.callback(Output('timeseries', 'figure'),
#              [Input('slider2', 'value')])
#def update_graph2(val):
#	print(val)


@app.callback(
    #Output('hover-data', 'children'),
    Output("hover-data", "data"),
    Input('timeseries', 'hoverData'))
def display_hover_data(hoverData):
	if hoverData is not None:
		#print("Did a callback")
		indices = []
		for p in hoverData['points']:
			custIndex = p['customdata'][0]
			indices.append(custIndex)

		df_selected = df[df.dummyKey.isin(indices)]
		#print(df_selected)

		#return json.dumps(hoverData, indent=2)
		return df_selected.to_dict('records')



@app.callback(
	#Output('selected-data', 'children'),
	[Output("selected-data", "data"),
	 Output('modelPredictionsGraph', 'figure'),
	 Output('mapeDiv', 'children')
	],
	Input('timeseries', 'selectedData'))
def display_selected_data(selectedData):
	if selectedData is not None:
		#print("Did a callback")
		indices = []
		for p in selectedData['points']:
			custIndex = p['customdata'][0]
			indices.append(custIndex)

		df_selected = df[df.dummyKey.isin(indices)]
		#print(df_selected)

		mape = mean_absolute_percentage_error(df_selected.BodyweightKg, df_selected.predicted)
		mapeString = "MAPE= {:.0%}".format(mape)


	figure = px.scatter(
		df_selected
		, x="BodyweightKg"
		, y=["predicted"]
		#, color="Continent"
		, custom_data = ['dummyKey']
		, trendline="ols"
		, title="Model - Weight vs predicted"
	)

	figure.update_layout({
		'plot_bgcolor': '#161616',
		'paper_bgcolor': '#161616',
		'template':'plotly_dark'
	})

	figure.update_layout(
		dragmode='select',
	)

	return df_selected.to_dict('records'), figure, mapeString





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
			#,  Output('WeightSliderRange', 'children')
			#,  Output('HeightSliderRange', 'children')
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


	# Update the slider value
	WeightSliderRangeValue = str(slider2)
	HeightSliderRangeValue = str(slider3)

	df_filtered = df.copy()	

	print(f"This is the input: {continentSelector}")
	if len(continentSelector) > 0:
		df_filtered = df_filtered[df_filtered.Continent.isin(continentSelector)]
	print(f"This is the input: {countrySelector}")
	if len(countrySelector) > 0:
		df_filtered = df_filtered[df_filtered.Country.isin(countrySelector)]
	print(f"This is the input: {eventSelector}")
	if len(eventSelector) > 0:
		df_filtered = df_filtered[df_filtered.Event.isin(eventSelector)]
	print(f"This is the input: {eventSelector}")
	if len(equipmentSelector) > 0:
		df_filtered = df_filtered[df_filtered.Equipment.isin(equipmentSelector)]
	print(f"This is the input: {parentFederationSelector}")
	if len(parentFederationSelector) > 0:
		df_filtered = df_filtered[df_filtered.ParentFederation.isin(parentFederationSelector)]
	print(f"This is the input: {validResSelector}")
	if len(validResSelector) > 0:
		filtering_list = []
		for item in validResSelector:
			if item == 'True':
				filtering_list.append(True)
			if item == 'False':
				filtering_list.append(False)
		df_filtered = df_filtered[df_filtered.Valid_Results.isin(filtering_list)]
	print(f"This is the input: {sexSelector}")
	if len(sexSelector) > 0:
		df_filtered = df_filtered[df_filtered.Sex.isin(sexSelector)]


	
	# Ranges
	df_filtered = df_filtered[df_filtered.BodyweightKg.fillna(0)>= slider2[0]]	
	df_filtered = df_filtered[df_filtered.BodyweightKg.fillna(0)<= slider2[1]]	

	df_filtered = df_filtered[df_filtered['Mean height'].fillna(0)>= slider3[0]]	
	df_filtered = df_filtered[df_filtered['Mean height'].fillna(0)<= slider3[1]]	
	#print(slider2[0], slider2[1])

	#print(df_filtered)

	rowsFound = df_filtered.shape[0]

	# Filter the dataset:
	## And subselect because of performance
	if nSamplesTotal > df_filtered.shape[0]:
		nSamplesTotal = df_filtered.shape[0]
	df_filtered = df_filtered.sample(nSamplesTotal, replace=False)
	rowsShown = df_filtered.shape[0]

	print(df_filtered.Valid_Results)


	# Refilter filters
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

	# Do some plotting
	print(continentSelector)

	
	figure = px.scatter(
        df_filtered
        , x="BodyweightKg"
        , y="TotalKg"
        , color="Continent"
        , custom_data = ['dummyKey']

        , title="Continent"
    )

	print(df_filtered.shape)


	print("HERE!")
	print(df_filtered.shape)
	# Do subsampling here!!!
	figure3 = px.density_contour(
        df_filtered
        , x=xAxisSelectorKDE
        , y=yAxisSelectorKDE
        , color=colorGroupingKDE
        , title="Distribution stuff"
        , marginal_x="histogram"
		, marginal_y="histogram"
    )

	df_grouped = df_filtered.groupby(['CompetitionYear'], as_index = False)[['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']].mean()
	#print(df_grouped)
	figure2 = px.area(
        df_grouped
        , x="CompetitionYear"
        , y=["Best3SquatKg", "Best3BenchKg", "Best3DeadliftKg"]
        #, color="Continent"
        #, color_continuous_scale=colorscale
        #, render_mode="webgl"
        , title="Average score over time"
	)
	
	figure.update_layout({
		'plot_bgcolor': '#161616',
		'paper_bgcolor': '#161616',
		'template':'plotly_dark'
		})

	figure.update_layout(
		dragmode='select',
	)

	figure2.update_layout({
		'plot_bgcolor': '#161616',
		'paper_bgcolor': '#161616',
		'template':'plotly_dark'
		})
	figure2.update_layout(
		dragmode='select',
	)

	figure3.update_layout({
		'plot_bgcolor': '#161616',
		'paper_bgcolor': '#161616',
		'template':'plotly_dark'
		})
	figure3.update_layout(
		dragmode='select',
	)
	

	print(df.shape)
	print(df_filtered.shape)


	if rowsFound > nSamplesTotal:
		counterText = 'Found ' + str(rowsFound) + ' competitors matching filter. Showing ' + str(rowsShown) + ' competitors'
	else:
		counterText = 'Found ' + str(rowsFound) + ' competitors matching filter.'
	print(counterText)

	return figure, figure2, figure3\
	, continent_options, country_options, event_options, equipment_options, parent_federation_options, valid_res_options, sex_options\
	, counterText
	#, WeightSliderRangeValue, HeightSliderRangeValue
	



if __name__ == '__main__':
    app.run_server(debug=True)