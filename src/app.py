import dash                                     
from dash import Dash, dcc, html, Input, Output
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import date
import pandas as pd

app = Dash(__name__,
           external_stylesheets=[
                dbc.themes.BOOTSTRAP, 
                dbc.icons.FONT_AWESOME
                ])
server = app.server

# Loading Data
conn_df = pd.read_csv('conn_df.csv')
conn_df['Trading_Date'] = pd.to_datetime(conn_df['Trading_Date'])

ori_df = pd.read_csv('ori_df.csv')
ori_df['Trading_Date'] = pd.to_datetime(ori_df['Trading_Date'])

sp_df = pd.read_csv('sp_df.csv')
sp_df['Trading_Date'] = pd.to_datetime(sp_df['Trading_Date'])

ems_df = pd.read_csv('ems_bat_df.csv')
ems_df['Trading_Date'] = pd.to_datetime(ems_df['Trading_Date'])

# Functions

def main_chart(total_daily):
    fig = make_subplots(rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=('Median Daily Spot Price of ISL2201 ($c/kWh)',
            'Daily Electricity Consumption (kWh)',
            ),
        vertical_spacing=0.1,
        row_width=[0.5,0.5])
    
    fig.add_trace(
        go.Scatter(x=total_daily['Trading_Date'], y=(total_daily['$c/kWh']), 
                    name='ISL2201 Median Spotprice ($c/kWh)',
                    hoverinfo='x+y',
                    mode='lines',
                    line=dict(width=1.5, color='rgba(135, 156, 66, 1)'),
                    hovertemplate = '%{y:,.2f} $c'),row=1,col=1
    )

    fig.add_trace(
        go.Scatter(x=total_daily['Trading_Date'], y=total_daily['Consumption(kWh)'], name='Consumption(kWh)',
            hoverinfo='x+y',
            mode='lines',
            line=dict(width=0.5, color='rgba(95, 178, 237, 1)'),
            stackgroup='two',
            hovertemplate = '%{y:,.2f} kWh'),row=2,col=1
    )

    # Add figure layout
    for n in range (2):
        fig.layout.annotations[n].update(x=0,font_size=14,xanchor ='left')
    fig.update_layout(title_text= 'Daily ISL2201 Average Spotprice, Electricity Consumption',
        height = 800,
        barmode = 'overlay',
        title_yanchor='top',
        hovermode="x unified",
        plot_bgcolor='#FFFFFF',
        margin = dict(r=20),
        xaxis = dict(tickmode = 'linear',dtick = 'M1'),
        legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1)
        )
    fig.update_yaxes(row=1, col=1, title='Price ($c/kWh)',showgrid=True, gridwidth=1, gridcolor='#f0f0f0',
                 title_font_size=12,tickfont=dict(size=12)
                 )
    fig.update_yaxes(row=2, col=1, title='Consumption(kWh)',showgrid=True, gridwidth=1, gridcolor='#f0f0f0',
                 title_font_size=12,tickfont=dict(size=12)
                 )

    fig.update_traces(xaxis='x2')
    fig.update_xaxes(showgrid=False, gridwidth=1, title_font_size=12,tickfont=dict(size=12), dtick='M1')

    return fig

def group_charts(total_df,clk_date):
    fig = make_subplots(rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=('ISL2201 Spot Price ($c/kWh)',
                        'Electricity Consumption (kWh)',
            # 'Net Electricity Consumption (kWh)',
            # 'Electricity Cost($)'
            ),
        vertical_spacing=0.1,
        row_width=[0.5,0.5])

    fig.add_trace(
        go.Scatter(x=total_df['Trading_Period'], y=(total_df['$c/kWh']), 
                    name='ISL2201 Spotprice ($c/kWh)',
                    hoverinfo='x+y',
                    mode='lines+markers',
                    line=dict(width=2, color='rgba(135, 156, 66,1)'),
                    hovertemplate = '%{y:,.2f} $c'),row=1,col=1
    )

    fig.add_trace(
        go.Scatter(x=total_df['Trading_Period'], y=total_df['Consumption(kWh)'], 
                    name='Consumption(kWh)',
                    hoverinfo='x+y',
                    mode='lines',
                    line=dict(width=0.5, color='rgba(95, 178, 237, 1)'),
                    stackgroup='two',
                    hovertemplate = '%{y:,.2f} kWh'),row=2,col=1
    )

    # Add figure layout
    for n in range (2):
        fig.layout.annotations[n].update(x=0,font_size=14,xanchor ='left')
    fig.update_xaxes(type='category', categoryorder='category ascending')
    fig.update_traces(xaxis='x2')
    fig.update_layout(title_text=(f'Electricity Consumption on {clk_date}'),
        title_yanchor='top',
        height = 800,
        hovermode="x unified",
        plot_bgcolor='#FFFFFF',
        barmode = 'overlay',
        margin = dict(r=20),
        # xaxis = dict(tickangle = -45, tickfont =dict(size=10),showticklabels=True),
        legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1,)
        )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0',
                     title_font_size=12,tickfont=dict(size=12))
    fig.update_yaxes(row = 1, col = 1, title='Consumption (kWh)')
    fig.update_yaxes(row = 2, col = 1, title='Spot Price ($c/kWh)')
    # fig.update_yaxes(row = 3, col = 1, title='Consumption (kWh)')
    # fig.update_yaxes(row = 4, col = 1, title='$NZ')
    fig.update_xaxes(showticklabels=True,tickangle= -60, showgrid=False, gridwidth=1, 
                     title_font_size=12,tickfont=dict(size=12))

    return fig

def daily_ems_charts(total_daily):

    fig = make_subplots(rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=('Carbon Intensity (g/kWh)',
            'Carbon Emission (ton)'),
        vertical_spacing=0.1,
        row_width=[0.6,0.4])
    
    fig.add_trace(
        go.Scatter(x=total_daily['Trading_Date'], y=total_daily['Carbon_Intensity(g/KWh)'], 
                    name='Carbon Intensity (g/kWh)',
                    hoverinfo='x+y',
                    mode='lines',
                    line=dict(width=1.5, dash='dot', color='rgba(116, 100, 175,1)'),
                    hovertemplate = '%{y:,.2f} g/kWh'),row=1,col=1
    )

    fig.add_trace(
        go.Bar(x=total_daily['Trading_Date'], y=total_daily['Carbon_Emission(t)'], 
                    name='Carbon Emission(t)',
                    marker=dict(color='rgba(183, 183, 183, 1)',line = dict(width=0)),
                    hovertemplate = '%{y:,.3f} ton'),row=2,col=1
    )

    # Add figure layout
    for n in range (2):
        fig.layout.annotations[n].update(x=0,font_size=14,xanchor ='left')
    fig.update_layout(title_text= 'Daily Average Carbon Intensity & Carbon Emission',
        height = 600,
        barmode = 'overlay',
        title_yanchor='top',
        hovermode="x unified",
        plot_bgcolor='#FFFFFF',
        margin = dict(r=20,t=170),
        xaxis = dict(tickmode = 'linear',dtick = 'M1'),
        legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1)
        )
    fig.update_yaxes(row=1, col=1, title='Carbon Intensity (g/kWh)',showgrid=True, gridwidth=1, gridcolor='#f0f0f0',
                 title_font_size=12,tickfont=dict(size=12)
                 )
    fig.update_yaxes(row=2, col=1, title='Carbon Emission(t)',showgrid=True, gridwidth=1, gridcolor='#f0f0f0',
                 title_font_size=12,tickfont=dict(size=12)
                 )
    fig.update_traces(xaxis='x2')
    fig.update_xaxes(showgrid=False, gridwidth=1, title_font_size=12,tickfont=dict(size=12), dtick='M1')
    return fig

def detail_ems_charts(total_df,clk_date):
    fig = make_subplots(rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=('Carbon Intensity (g/kWh)',
            'Carbon Emission with and without Solar (kg)'),
        vertical_spacing=0.1,
        row_width=[0.6,0.4])
    
    fig.add_trace(
        go.Scatter(x=total_df['Trading_Period'], y=total_df['Carbon_Intensity(g/KWh)'], 
                    name='Carbon Intensity (g/kWh)',
                    hoverinfo='x+y',
                    mode='lines+markers',
                    line=dict(width=1.5, dash='dash', color='rgba(116, 100, 175,1)'),
                    hovertemplate = '%{y:,.2f} g/kWh'),row=1,col=1
    )

    fig.add_trace(
        go.Bar(x=total_df['Trading_Period'], y=total_df['Carbon_Emission(kg)'], 
                    name='Carbon Emission (kg)',
                    marker=dict(color='rgba(183, 183, 183, 1)',line = dict(width=0)),
                    hovertemplate = '%{y:,.3f} kg'),row=2,col=1
    )

    # Add figure layout
    for n in range (2):
        fig.layout.annotations[n].update(x=0,font_size=14,xanchor ='left')
    fig.update_layout(title_text= f'Carbon Intensity & Carbon Emission from Isaac power consumption on {clk_date}',
        height = 600,
        barmode = 'overlay',
        title_yanchor='top',
        hovermode="x unified",
        plot_bgcolor='#FFFFFF',
        margin = dict(r=20,t=170),
        xaxis = dict(tickmode = 'linear',dtick = 'M1'),
        legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1)
        )
    fig.update_yaxes(row=1, col=1, title='Carbon Intensity (g/kWh)',showgrid=True, gridwidth=1, gridcolor='#f0f0f0',
                 title_font_size=12,tickfont=dict(size=12)
                 )
    fig.update_yaxes(row=2, col=1, title='Carbon Emission (kg)',showgrid=True, gridwidth=1, gridcolor='#f0f0f0',
                 title_font_size=12,tickfont=dict(size=12)
                 )
    fig.update_traces(xaxis='x2')
    fig.update_xaxes(showgrid=False, gridwidth=1, title_font_size=12,tickfont=dict(size=12), dtick='M1')
    return fig


# Layout
tab1 = html.Div([
    html.Br(),html.Br(),
    html.H3('CONN. SITE '),
    html.Br(),
    html.H3('Electricity Consumption & Spot Price Charts'),
    html.Br(),
    dbc.Row(html.Div(id='summary',children={})),
    html.Br(),
    dbc.Row([
        dcc.Loading(id='main_loading',children=
                    [dbc.Card(dcc.Graph(id='overview-graph', figure={},
                                        clickData=None, 
                                        hoverData=None))],
            ),
    ]),
    html.Br(),
    dbc.Row([  
        dcc.Loading(id='detail_loading',children=
            [dbc.Card(dcc.Graph(id='detail-graph', figure={}, clickData=None, 
                hoverData=None)
            )],
            type = 'default',
        )
    ]),
    html.Br(),html.Br(),html.Br(),
    html.H3('Carbon Intensity and Carbon Emission Charts'),
    html.Br(),
    dbc.Row(html.Div(id='ems-summary',children={})),
    html.Br(),
    dbc.Row([  
        dcc.Loading(id='ems_loading',children=
            [dbc.Card(dcc.Graph(id='ems-graph', figure={}, clickData=None, 
                hoverData=None)
            )],
            type = 'default',
        )
    ]),
    html.Br(),
    dbc.Row([  
        dcc.Loading(id='ems_detail_loading',children=
            [dbc.Card(dcc.Graph(id='ems-detail-graph', figure={}, clickData=None, 
                hoverData=None)
            )],
            type = 'default',
        )
    ]),
])

tab2 = html.Div([
    html.Br(),html.Br(),
    html.H3('ORI WAIRAKEI SITE '),
    html.Br(),
    html.H3('Electricity Consumption & Spot Price Charts'),
    html.Br(),
    dbc.Row(html.Div(id='summary1',children={})),
    html.Br(),
    dbc.Row([
        dcc.Loading(id='main_loading1',children=
                    [dbc.Card(dcc.Graph(id='overview-graph1', figure={},
                                        clickData=None, 
                                        hoverData=None))],
            ),
    ]),
    html.Br(),
    dbc.Row([  
        dcc.Loading(id='detail_loading1',children=
            [dbc.Card(dcc.Graph(id='detail-graph1', figure={}, clickData=None, 
                hoverData=None)
            )],
            type = 'default',
        )
    ]),
    html.Br(),html.Br(),html.Br(),
    html.H3('Carbon Intensity and Carbon Emission Charts'),
    html.Br(),
    dbc.Row(html.Div(id='ems-summary1',children={})),
    html.Br(),
    dbc.Row([  
        dcc.Loading(id='ems_loading1',children=
            [dbc.Card(dcc.Graph(id='ems-graph1', figure={}, clickData=None, 
                hoverData=None)
            )],
            type = 'default',
        )
    ]),
    html.Br(),
    dbc.Row([  
        dcc.Loading(id='ems_detail_loading1',children=
            [dbc.Card(dcc.Graph(id='ems-detail-graph1', figure={}, clickData=None, 
                hoverData=None)
            )],
            type = 'default',
        )
    ]),
])

app.layout = dbc.Container([
    html.H2("Battery Test Sides Visualisation", style={'font-family':'arial','textAlign':'center'}),
    html.Br(),html.Br(),
       
    dcc.Tabs(id="tabs-charts", value='tab-1', children=[
        dcc.Tab(label='CONN. SITE', value='tab-1'),
        dcc.Tab(label='ORI WAIRAKEI SITE', value='tab-2'),
    ]),
    html.Div(id='tabs-content')
],style={ 'padding':'15px'},fluid=True)


# Callback
@app.callback(Output('tabs-content', 'children'),
              Input('tabs-charts', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return tab1
    elif tab == 'tab-2':
        return tab2

# Tab 1
@app.callback(
    Output(component_id='overview-graph', component_property='figure'),
    Input(component_id='tabs-charts', component_property='value'),
    # Input(component_id='input_size2', component_property='value')
)
def update_mainchart(tab):
    spotprice_df = (sp_df[(sp_df['Trading_Date']<=conn_df['Trading_Date'].max()) & 
                          (sp_df['Trading_Date']>=conn_df['Trading_Date'].min())
                          ])
    emission_df = (ems_df[(ems_df['Trading_Date']<=conn_df['Trading_Date'].max()) & 
                          (ems_df['Trading_Date']>=conn_df['Trading_Date'].min())
                          ])
    
    total_df = (conn_df.merge(spotprice_df, on = ['Trading_Date','Trading_Period'])
                # .merge(soldf2, on = ['Trading_Date','Trading_Period'])
                .merge(emission_df, on =['Trading_Date','Trading_Period'])
                )

    total_daily = (total_df.groupby('Trading_Date',as_index=False)
                   .agg({'Consumption(kWh)':'sum','$/MWh':'median'}))
    total_daily['$c/kWh'] = round(total_daily['$/MWh']/10,2)

    fig = main_chart(total_daily)

    return fig

@app.callback(
    Output(component_id='detail-graph', component_property='figure'),
    # Input(component_id='input_size', component_property='value'),
    # Input(component_id='input_size2', component_property='value'),
    Input(component_id='overview-graph', component_property='clickData')
)
def update_group_charts(clk_data):
    spotprice_df = (sp_df[(sp_df['Trading_Date']<=conn_df['Trading_Date'].max()) & 
                          (sp_df['Trading_Date']>=conn_df['Trading_Date'].min())
                          ])
    emission_df = (ems_df[(ems_df['Trading_Date']<=conn_df['Trading_Date'].max()) & 
                          (ems_df['Trading_Date']>=conn_df['Trading_Date'].min())
                          ])
    
    total_df = (conn_df.merge(spotprice_df, on = ['Trading_Date','Trading_Period'])
                # .merge(soldf2, on = ['Trading_Date','Trading_Period'])
                .merge(emission_df, on =['Trading_Date','Trading_Period'])
                )
    

    if clk_data is None:
        clk_date = total_df['Trading_Date'].min()
        df = total_df[total_df['Trading_Date'] == clk_date]
        df['$c/kWh'] = round(df['$/MWh']/10,2)
        fig2 = group_charts(df,clk_date)

        return fig2
    else:

        clk_date = clk_data['points'][0]['x']
        df = total_df[total_df['Trading_Date'] == clk_date]
        df['$c/kWh'] = round(df['$/MWh']/10,2)
        fig2 = group_charts(df,clk_date)
        
        return fig2

@app.callback(
    Output(component_id='ems-graph', component_property='figure'),
    Input(component_id='tabs-charts', component_property='value'),
    # Input(component_id='input_size', component_property='value'),
    # Input(component_id='input_size2', component_property='value')
)
def update_emschart(tab):
    spotprice_df = (sp_df[(sp_df['Trading_Date']<=conn_df['Trading_Date'].max()) & 
                          (sp_df['Trading_Date']>=conn_df['Trading_Date'].min())
                          ])
    emission_df = (ems_df[(ems_df['Trading_Date']<=conn_df['Trading_Date'].max()) & 
                          (ems_df['Trading_Date']>=conn_df['Trading_Date'].min())
                          ])
    
    total_df = (conn_df.merge(spotprice_df, on = ['Trading_Date','Trading_Period'])
                # .merge(soldf2, on = ['Trading_Date','Trading_Period'])
                .merge(emission_df, on =['Trading_Date','Trading_Period'])
                )
    total_df['Carbon_Emission(t)'] = (
        round(total_df['Consumption(kWh)']*total_df['Carbon_Intensity(g/KWh)']/1000000,3))

    total_daily = total_df.groupby('Trading_Date',as_index=False).sum(numeric_only=True)
    total_daily['$c/kWh'] = round(total_daily['$/MWh']/10,2)

    fig = daily_ems_charts(total_daily)

    return fig        

@app.callback(
    Output(component_id='ems-detail-graph', component_property='figure'),
    # Input(component_id='input_size', component_property='value'),
    # Input(component_id='input_size2', component_property='value'),
    Input(component_id='ems-graph', component_property='clickData'),
)
def update_emsdetailchart(clk_data):
    spotprice_df = (sp_df[(sp_df['Trading_Date']<=conn_df['Trading_Date'].max()) & 
                          (sp_df['Trading_Date']>=conn_df['Trading_Date'].min())
                          ])
    emission_df = (ems_df[(ems_df['Trading_Date']<=conn_df['Trading_Date'].max()) & 
                          (ems_df['Trading_Date']>=conn_df['Trading_Date'].min())
                          ])
    
    total_df = (conn_df.merge(spotprice_df, on = ['Trading_Date','Trading_Period'])
                # .merge(soldf2, on = ['Trading_Date','Trading_Period'])
                .merge(emission_df, on =['Trading_Date','Trading_Period'])
                )
    total_df['Carbon_Emission(kg)'] = (
        round(total_df['Consumption(kWh)']*total_df['Carbon_Intensity(g/KWh)']/1000,3))

    if clk_data is None:
        clk_date = total_df['Trading_Date'].min()
        detail_ems = total_df[total_df['Trading_Date'] == clk_date]
        fig2 = detail_ems_charts(detail_ems,clk_date)
        return fig2
    
    else:
        clk_date = clk_data['points'][0]['x']
        detail_ems = total_df[total_df['Trading_Date'] == clk_date]
        fig2 = detail_ems_charts(detail_ems,clk_date)
        return fig2

# Tab 2
@app.callback(
    Output(component_id='overview-graph1', component_property='figure'),
    Input(component_id='tabs-charts', component_property='value'),
    # Input(component_id='input_size2', component_property='value')
)
def update_mainchart1(tab):
    spotprice_df = (sp_df[(sp_df['Trading_Date']<=ori_df['Trading_Date'].max()) & 
                          (sp_df['Trading_Date']>=ori_df['Trading_Date'].min())
                          ])
    emission_df = (ems_df[(ems_df['Trading_Date']<=ori_df['Trading_Date'].max()) & 
                          (ems_df['Trading_Date']>=ori_df['Trading_Date'].min())
                          ])
    
    total_df = (ori_df.merge(spotprice_df, on = ['Trading_Date','Trading_Period'])
                # .merge(soldf2, on = ['Trading_Date','Trading_Period'])
                .merge(emission_df, on =['Trading_Date','Trading_Period'])
                )

    total_daily = (total_df.groupby('Trading_Date',as_index=False)
                   .agg({'Consumption(kWh)':'sum','$/MWh':'median'}))
    total_daily['$c/kWh'] = round(total_daily['$/MWh']/10,2)

    fig = main_chart(total_daily)

    return fig

@app.callback(
    Output(component_id='detail-graph1', component_property='figure'),
    # Input(component_id='input_size', component_property='value'),
    # Input(component_id='input_size2', component_property='value'),
    Input(component_id='overview-graph1', component_property='clickData')
)
def update_group_charts1(clk_data):
    spotprice_df = (sp_df[(sp_df['Trading_Date']<=ori_df['Trading_Date'].max()) & 
                          (sp_df['Trading_Date']>=ori_df['Trading_Date'].min())
                          ])
    emission_df = (ems_df[(ems_df['Trading_Date']<=ori_df['Trading_Date'].max()) & 
                          (ems_df['Trading_Date']>=ori_df['Trading_Date'].min())
                          ])
    
    total_df = (ori_df.merge(spotprice_df, on = ['Trading_Date','Trading_Period'])
                # .merge(soldf2, on = ['Trading_Date','Trading_Period'])
                .merge(emission_df, on =['Trading_Date','Trading_Period'])
                )
    

    if clk_data is None:
        clk_date = total_df['Trading_Date'].min()
        df = total_df[total_df['Trading_Date'] == clk_date]
        df['$c/kWh'] = round(df['$/MWh']/10,2)
        fig2 = group_charts(df,clk_date)

        return fig2
    else:

        clk_date = clk_data['points'][0]['x']
        df = total_df[total_df['Trading_Date'] == clk_date]
        df['$c/kWh'] = round(df['$/MWh']/10,2)
        fig2 = group_charts(df,clk_date)
        
        return fig2

@app.callback(
    Output(component_id='ems-graph1', component_property='figure'),
    Input(component_id='tabs-charts', component_property='value'),
    # Input(component_id='input_size', component_property='value'),
    # Input(component_id='input_size2', component_property='value')
)
def update_emschart1(tab):
    spotprice_df = (sp_df[(sp_df['Trading_Date']<=ori_df['Trading_Date'].max()) & 
                          (sp_df['Trading_Date']>=ori_df['Trading_Date'].min())
                          ])
    emission_df = (ems_df[(ems_df['Trading_Date']<=ori_df['Trading_Date'].max()) & 
                          (ems_df['Trading_Date']>=ori_df['Trading_Date'].min())
                          ])
    
    total_df = (ori_df.merge(spotprice_df, on = ['Trading_Date','Trading_Period'])
                # .merge(soldf2, on = ['Trading_Date','Trading_Period'])
                .merge(emission_df, on =['Trading_Date','Trading_Period'])
                )
    total_df['Carbon_Emission(t)'] = (
        round(total_df['Consumption(kWh)']*total_df['Carbon_Intensity(g/KWh)']/1000000,3))

    total_daily = total_df.groupby('Trading_Date',as_index=False).sum(numeric_only=True)
    total_daily['$c/kWh'] = round(total_daily['$/MWh']/10,2)

    fig = daily_ems_charts(total_daily)

    return fig        

@app.callback(
    Output(component_id='ems-detail-graph1', component_property='figure'),
    # Input(component_id='input_size', component_property='value'),
    # Input(component_id='input_size2', component_property='value'),
    Input(component_id='ems-graph1', component_property='clickData'),
)
def update_emsdetailchart1(clk_data):
    spotprice_df = (sp_df[(sp_df['Trading_Date']<=ori_df['Trading_Date'].max()) & 
                          (sp_df['Trading_Date']>=ori_df['Trading_Date'].min())
                          ])
    emission_df = (ems_df[(ems_df['Trading_Date']<=ori_df['Trading_Date'].max()) & 
                          (ems_df['Trading_Date']>=ori_df['Trading_Date'].min())
                          ])
    
    total_df = (ori_df.merge(spotprice_df, on = ['Trading_Date','Trading_Period'])
                # .merge(soldf2, on = ['Trading_Date','Trading_Period'])
                .merge(emission_df, on =['Trading_Date','Trading_Period'])
                )
    total_df['Carbon_Emission(kg)'] = (
        round(total_df['Consumption(kWh)']*total_df['Carbon_Intensity(g/KWh)']/1000,3))

    if clk_data is None:
        clk_date = total_df['Trading_Date'].min()
        detail_ems = total_df[total_df['Trading_Date'] == clk_date]
        fig2 = detail_ems_charts(detail_ems,clk_date)
        return fig2
    
    else:
        clk_date = clk_data['points'][0]['x']
        detail_ems = total_df[total_df['Trading_Date'] == clk_date]
        fig2 = detail_ems_charts(detail_ems,clk_date)
        return fig2


if __name__ == "__main__":
    app.run_server(debug=False)