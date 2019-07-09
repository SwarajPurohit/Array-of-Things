library(ggplot2)
library(magrittr)
library(dplyr)
library(maptools)
library(shiny)
library(rgeos)
library(mapproj)
library(RColorBrewer)
library(scales)
library(ggmap)
library(tensorflow)
library(plotly)
## tensorflow install
#library("devtools")
#devtools::install_github("rstudio/tensorflow")
#library(tensorflow)
#install_tensorflow()
#======================================================================================================
# 1. LOAD AND PRE- PROCESS DATA.                                                                      #
#======================================================================================================
# Load sensor data csv file. Fix data types and incorrect unit for ug/m^3.
df <- read.csv("final.csv")

df["timestamp"] <- lapply(df["timestamp"], function(x) as.POSIXct(strptime(x, "%d/%m/%Y %H:%M")))
df["parameter"] <- lapply(df["parameter"], function(x) toupper(x))

df$hrf_unit <- as.character(df$hrf_unit)                                                               
df$hrf_unit <- replace(df$hrf_unit, df$hrf_unit == "ÃŽÂ¼g/m^3", "ug/m^3")

# Define pollution limits
pol_standards <- list("CO"=35, "NO2"=0.1, "O3"=0.07, "SO2"=0.5, "PM10"=150, "PM2_5"=35, "PM1"=35)

# Load shapefile. 
shape <- readShapeSpatial("./ComFiles/ChiComArea")
shape_data <- fortify(shape, region = "ComAreaID") #Convert shapefile to dataframe

id <- shape$ComAreaID
com <- shape$community

shape_data <- merge(shape_data, data.frame(id, com), by='id') #Get community names along with id

latest <- subset(df, timestamp == df[length(df),1], ) # Get latest data for current map
#======================================================================================================
# 2. FUNCTION TO CREATE PAST TRENDS PLOT.                                                                #
#======================================================================================================
# Aggregate all if 'overall' is selected else groupby selected area. Plot limit line only for pollutants.

past_lookup <- function(area, param){
  
  if (area == "OVERALL"){
    req_df <- df %>% group_by(timestamp, parameter, hrf_unit) %>% summarise(value_hrf = mean(value_hrf)) %>% subset(parameter == param, c(timestamp, value_hrf, hrf_unit))    
    unit <- as.character(unlist(req_df[1,3]))  
    plot <- ggplot(req_df, aes(x = timestamp, y=value_hrf, group=group)) + geom_line(aes(group=1),col='blue') + ylab(unit) + ggtitle(paste("PAST TREND OF",param,"IN CHICAGO (overall)"))
    
    if (param %in% c("CO", "NO2", "O3", "SO2", "PM10", "PM2_5", "PM1")){
      
      plot <- plot + geom_hline(yintercept=get(param, pol_standards), linetype="dashed", color="red") + annotate("text", label = "Permissible limit", x = unlist(df[500,1]),y = get(param, pol_standards))
      ggplotly(plot)
    }else{
      ggplotly(plot)
    }          
  }
  
  else{
    req_df <- df %>% group_by(timestamp, com, parameter, hrf_unit) %>% summarise(value_hrf = mean(value_hrf)) %>% subset(com == area & parameter == param, c(timestamp, value_hrf, hrf_unit))    
    unit <- as.character(unlist(req_df[1,3]))
    plot <- ggplot(req_df, aes(x = timestamp, y=value_hrf, group=group)) + geom_line(aes(group=1),col='blue') + ylab(unit) + ggtitle(paste("PAST TREND OF",param,"IN",area))
    
    if (param %in% c("CO", "NO2", "O3", "SO2", "PM10", "PM2_5", "PM1")){
      
      plot<- plot + geom_hline(yintercept=get(param, pol_standards), linetype="dashed", color="red") + annotate("text", label = "Permissible limit", x = unlist(df[500,1]),y = get(param, pol_standards))
      ggplotly(plot)
    }else{
      ggplotly(plot)
    }     
  }
}
#======================================================================================================
# 3. FUNCTION TO SHOW MAP OF CURRENT DATA.
#======================================================================================================

curr_lookup <- function(param){
  
  req_df <- latest %>% group_by(com, parameter, hrf_unit) %>% summarise(value_hrf = mean(value_hrf)) %>% subset(parameter == param, c(com, value_hrf, hrf_unit))
  unit <- as.character(unlist(req_df[1,3]))
  
  final <- merge(shape_data, req_df, by='com', all.x=TRUE)
  final <- final[order(final$order), ]
  
  cnames <- aggregate(cbind(long, lat) ~ com, data=final, FUN=function(x) mean(range(x)))
  
  plot <- ggplot() + geom_polygon(data = final, aes(x = long, y = lat, group = group, fill = value_hrf), color = "black", size = 0.25)+ 
    coord_map() + scale_fill_distiller(name=unit, palette = "YlOrBr", direction=1, breaks = pretty_breaks(n = 5))+
    theme_nothing(legend = TRUE) + labs(title=paste("CURRENT LEVELS OF",param))+
    geom_text(data=cnames, aes(long, lat, label = com), size=1.8, fontface="bold")
  
  plot <- ggplotly(plot)
  plot%>%style(hoverinfo = 'text', traces = seq(1.50))
  
}

avg_finder <- function(param){
  req_df <- latest %>% group_by(com, parameter, hrf_unit) %>% summarise(value_hrf = mean(value_hrf)) %>% subset(parameter == param, c(com, value_hrf, hrf_unit))
  unit <- as.character(unlist(req_df[1,3]))
  avg <- mean(req_df$value_hrf)
  if (param %in% c("CO", "NO2", "O3", "SO2", "PM10", "PM2_5", "PM1")){
    if(avg > get(param, pol_standards)){
      return(paste("Average: <font color=\"#FF0000\"><b>", avg, "</b></font>",unit))
    }else{
      return(paste("Average: <font color=\"#008000\"><b>", avg, "</b></font>", unit))
    }
  }else{
    return(paste("Average: ",avg,unit))
  }
}
#======================================================================================================
# 4. FORECAST SO2 WITH MODEL
#======================================================================================================
so2.df <- read.csv("so2_ts.csv")

so2.vals <- so2.df$value_hrf
so2.vals_scaled <- (so2.vals -min(so2.vals))/(max(so2.vals)-min(so2.vals)) #min max scaling

# Define the tensorflow graph
tf$reset_default_graph()

num_inputs = 1
num_time_steps = 75
num_neurons = 200
num_outputs = 1
learning_rate = 0.0001
batch_size = 1

X <- tf$placeholder(tf$float32, c(1, num_time_steps, num_inputs))
y <- tf$placeholder(tf$float32, c(1, num_time_steps, num_outputs))

cell <- tf$contrib$rnn$OutputProjectionWrapper(
  tf$contrib$rnn$LSTMCell(num_units=num_neurons, activation=tf$nn$relu), 
  output_size=num_outputs) 

out <- tf$nn$dynamic_rnn(cell, X, dtype=tf$float32, parallel_iterations= as.integer(256))
outputs <- out[1]

loss <- tf$reduce_mean(tf$square(outputs - y)) # MSE

optimizer <- tf$train$RMSPropOptimizer(learning_rate=learning_rate,momentum=0.9,centered=TRUE,decay=0.9)
train <- optimizer$minimize(loss)

init <- tf$global_variables_initializer()

saver <- tf$train$Saver()

#Load the model and forecast
with(tf$Session() %as% sess,{
  
  saver$restore(sess, "./SO2 model/mod_so2.ckpt")
  
  seed = so2.vals_scaled[(length(so2.vals_scaled)-num_time_steps+1):length(so2.vals_scaled)]
  
  for (iteration in 1:289){
    
    X_batch = array_reshape(seed[(length(seed)-num_time_steps+1):length(seed)], c(1, num_time_steps, 1))
    y_pred = sess$run(outputs, feed_dict= dict(X = X_batch))
    y_pred = array_reshape(y_pred, c(1,num_time_steps,1))
    seed <- c(seed, y_pred[1,num_time_steps,1])
  }
})

so2.preds <- seed[(num_time_steps+1):length(seed)]
so2.preds*(max(so2.vals)-min(so2.vals))+min(so2.vals) #UNscale
future_ts <- seq.POSIXt(df[length(df),1], df[length(df),1]+86400, by = "5 min") #Create timeseries vector for the predictions
#======================================================================================================
# 5.Forecast plot function
#======================================================================================================
future_lookup <- function(param){
  if (param == "SO2"){
    plot <- ggplot(data.frame(future_ts,so2.preds), aes(x = future_ts, y=so2.preds, group=group)) + geom_line(aes(group=1),col='blue') + ylab("ppm") + xlab("timestamp")+ ggtitle(paste("FORECAST FOR",param))
    plot <- plot + geom_hline(yintercept=get(param, pol_standards), linetype="dashed", color="red") + annotate("text", label = "Permissible limit", x = unlist(future_ts[50]),y = get(param, pol_standards))
    ggplotly(plot)  
  }
}

#======================================================================================================
# 6. SHINY UI
#======================================================================================================
ui <- fluidPage(
  tabsetPanel(
    tabPanel("Past Trend", fluid = TRUE,
             sidebarLayout(
               sidebarPanel(selectInput("param_past", "Select Parameter:", choices=as.vector(unique(df$parameter))),
                            selectInput("area", "Select Community:", choices=NULL)),
               mainPanel(
                 plotlyOutput("plot_past")
               )
             )
    ),
    
    tabPanel("Current", fluid = TRUE,
             sidebarLayout(
               sidebarPanel(selectInput("param_curr", "Select Parameter:", choices=as.vector(unique(df$parameter))),
                htmlOutput("avg_curr")
               ),
               
               mainPanel(
                 plotlyOutput("plot_curr",width = "100%", height = "850px")
               )
             )
    ),
    
    tabPanel("Forecast", fluid = TRUE,
             sidebarLayout(
               sidebarPanel(selectInput("param_fore", "Select Parameter:", choices=c("SO2"))
               ),
               
               mainPanel(
                 plotlyOutput("plot_fore")
               )
             )
    ) 
  )
)

#======================================================================================================
# 7. SHINY SERVER
#======================================================================================================
server <- function(input, output, session){
  
  observe({
    x <- as.vector(unique(df[df['parameter'] == input$param_past,8]))
    updateSelectInput(session, "area", "Select Community:", choices=c("OVERALL",x))      
  })
  
  output$plot_past <- renderPlotly({
    past_lookup(input$area, input$param_past)
  })
  
  output$plot_curr <- renderPlotly({
    curr_lookup(input$param_curr)
  })
  
  output$plot_fore <- renderPlotly({
    future_lookup(input$param_fore)
  })
  output$avg_curr <- renderText({
    avg_finder(input$param_curr)
  })
  
  session$onSessionEnded(stopApp)  
}
#======================================================================================================
shinyApp(ui, server)
