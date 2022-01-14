## Imports
import pandas as pd
import numpy as np
import json

from sklearn.metrics import mean_absolute_percentage_error

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import dash
from dash import dcc
from dash import html
import dash_table as dt
from dash.dependencies import Input, Output# Load Data

# Read in the data
#df = pd.read_csv('dataframe_to_visualize_subset.csv', index_col=0, parse_dates=True)
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

# Initialize dropdowns to all values, before callback is called first time.
## Enables selective filtering
continent_options_all         = get_options(df['Continent'].unique())
country_options_all           = get_options(df['Country'].unique())
event_options_all             = get_options(df['Event'].unique())
equipment_options_all         = get_options(df['Equipment'].unique())
parent_federation_options_all = get_options(df['ParentFederation'].unique())
valid_res_options_all         = get_options(df['Valid_Results'].unique())
sex_options_all               = get_options(df['Sex'].unique())

continent_options         = continent_options_all
country_options           = country_options_all
event_options             = event_options_all
equipment_options         = equipment_options_all
parent_federation_options = parent_federation_options_all
valid_res_options         = valid_res_options_all
sex_options               = sex_options_all 

print("Generated dropdown options")

# Headers used for exploratory tables
tableColumns = ['Name', 'Country', 'Sex', 'Age', 'BodyweightKg', 'Date', 'Best3SquatKg','Best3BenchKg', 'Best3DeadliftKg', 'TotalKg', 'Wilks','Mean height', 'predicted']
tableColumns = ['Name', 'Country', 'Sex', 'Age', 'BodyweightKg', 'predicted', 'TotalKg', 'Mean height']

# Prettify printing (Don't think this worked)
pd.options.display.float_format = '${:.2f}'.format

# Get ranges for sliders
date_min = df['Date'].min()
date_max = df['Date'].max()

bodyweight_min = df['BodyweightKg'].min()
bodyweight_max = df['BodyweightKg'].max()

height_min = df['Mean height'].min()
height_max = df['Mean height'].max()

print("Generated slider options")

# Generate extra options
drilldownfilter_options = get_options(['No', 'Yes'])

# Options for KDE plot is handled separately, as they don't get updated
axis_options_kde = get_options(['BodyweightKg', 'Mean height', 'TotalKg', 'Dots', 'Wilks', 'Age', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'Glossbrenner', 'Goodlift', 'predicted', 'CompetitionYear'])
grouping_options_kde = get_options(['Sex', 'Continent', 'ParentFederation', 'Equipment', 'Country', 'Event'])

# Define layout of the page 
app.layout = html.Div(
    children = [
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
			                                                    , style={'backgroundColor': '#1E1E00'}
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
																	included=True,
																	allowCross=False,
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
	[Output("selected-data", "data")
	, Output('modelPredictionsGraph', 'figure')
	, Output('mapeDiv', 'children')]
	, Input('timeseries', 'selectedData')
)
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

		# Restructure data for 3D plotting - Needs some work as I want each plot (predicted vs actual) separted
		## Define values
		x  = df_selected['Mean height'].values
		y1 = df_selected['BodyweightKg'].values
		y2 = df_selected['predicted'].values
		z  = df_selected['TotalKg'].values

		# Create trace for Actual
		trace1 = go.Scatter3d(x = x
							, y = y1
							, z = z
							, mode = 'markers'
							, marker = dict(color="blue")
							, name = 'BW (True)'
		)
		
		# Create trace for predicted
		trace2 = go.Scatter3d(x = x
							, y = y2
							, z = z
							, mode = 'markers'
							, marker = dict(color="red")
							, name = 'BW (Pred)'
		)

		# Link the points with lines, for easier exploration where each prediction went

		# List of points
		x_lines = list()
		y_lines = list()
		z_lines = list()

		# Concatenate both 
		## Each point gets [from, to, None], where "None" ensures that they don't get connected into a snake
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

		# Trace the lines generated
		trace3 = go.Scatter3d(x = x_lines
							, y = y_lines
							, z = z_lines
							, mode = 'lines'
							, name = ""
		)

		# Create figure from the three traces
		fig = go.Figure(data = [trace1, trace2, trace3])

		# update legend
		fig.update_layout(legend = dict(orientation = "h"
									, yanchor = "bottom"
									, y = 1.02
									, xanchor = "right"
									, x = 1
									)
		)

		# update axis
		fig.update_layout(scene = dict(xaxis = dict(title_text='Mean height (cm)')
									,  yaxis = dict(title_text='Bodywieght (kg)')
									,  zaxis = dict(title_text='Total (kg)')
									)
		)

		# Update layout
		fig.update_layout({'plot_bgcolor': '#161616'
						,  'paper_bgcolor': '#161616'
						,  'template': 'plotly_dark'
						}
		)

		# Over and out
		d = df_selected.to_dict('records')
	
	#If no point is selected, remove all
	else: 
		d = df.head(0).to_dict('records')
		fig = go.Figure().add_annotation( x = 2
										, y = 2
										, text = "No Data to Display"
										, font = dict(family="sans serif"
													, size=25
													, color="crimson"
												)
										, showarrow = False
										, yshift = 10
		)
		mapeString = "Nothing Selected"
	# Return stuff
	return d, fig, mapeString
	
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
		continent_options         = continent_options_all
		country_options           = country_options_all
		event_options             = event_options_all
		equipment_options         = equipment_options_all
		parent_federation_options = parent_federation_options_all
		valid_res_options         = valid_res_options_all
		sex_options               = sex_options_all
	else:
		country_options           = get_options(df_filtered['Country'].unique())
		continent_options         = get_options(df_filtered['Continent'].unique())
		country_options           = get_options(df_filtered['Country'].unique())
		event_options             = get_options(df_filtered['Event'].unique())
		equipment_options         = get_options(df_filtered['Equipment'].unique())
		parent_federation_options = get_options(df_filtered['ParentFederation'].unique())
		valid_res_options         = get_options(df_filtered['Valid_Results'].unique())
		sex_options               = get_options(df_filtered['Sex'].unique())

	# Create the plots	

	## Standard scatterplot
	figure = px.scatter(  df_filtered
						, x="BodyweightKg"
        				, y="TotalKg"
        				, color="Continent"
        				# Append key for filtering
        				, custom_data = ['dummyKey']
        				, title="Continent"
    )

	## Update layout
	figure.update_layout({'plot_bgcolor': '#161616'
						, 'paper_bgcolor': '#161616'
						, 'template':'plotly_dark'
						}
	)

    ## Default action
	figure.update_layout(dragmode='select')

	# KDE Plot - based on selectors
	figure3 = px.density_contour( df_filtered
						        , x=xAxisSelectorKDE
						        , y=yAxisSelectorKDE
						        , color=colorGroupingKDE
						        , title="Distribution stuff"
						        , marginal_x="histogram"
								, marginal_y="histogram"
    )

	## Update layout
	figure3.update_layout({'plot_bgcolor': '#161616'
						,  'paper_bgcolor': '#161616'
						,  'template':'plotly_dark'
						}
	)
    ## Default action
	figure3.update_layout(dragmode='select')

	# Do aggregation
	df_grouped = df_filtered.groupby(['CompetitionYear'], as_index = False)[['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']].mean()
	
	# Show timeline
	figure2 = px.area(df_grouped
        			, x="CompetitionYear"
        			, y=["Best3SquatKg", "Best3BenchKg", "Best3DeadliftKg"]
        			, title="Average score over time"
	)
	
	## Update layout
	figure2.update_layout({'plot_bgcolor': '#161616'
						,  'paper_bgcolor': '#161616'
						,  'template':'plotly_dark'
						}
	)
	## Default action
	figure2.update_layout(dragmode='select')

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