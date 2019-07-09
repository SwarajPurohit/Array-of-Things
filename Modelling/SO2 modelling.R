library(tensorflow)

df <- read.csv("so2_ts.csv")

#================================================================
next_batch <- function(training_data,batch_size,steps){
  
  rand_start <- sample(1:(length(training_data)-steps-1), 1)
  
  
  data =training_data[rand_start:(rand_start+steps)]
  
  batch <- list("X"=array_reshape(data[1:(length(data)-1)], c(-1,steps,1)), "y"=array_reshape(data[2:length(data)], c(-1,steps,1)))
  return(batch)
}


#================================================================

vals <- df$value_hrf
vals_scaled <- (vals -min(vals))/(max(vals)-min(vals))

train_set <- vals_scaled[0:1942]
test_set <- vals_scaled[1943:1992]
#================================================================

tf$reset_default_graph()


num_inputs = 1
num_time_steps = 75
num_neurons = 200
num_outputs = 1
learning_rate = 0.0001
num_train_iterations = 100000
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

#================================================

with(tf$Session(config=tf$ConfigProto(allow_soft_placement=TRUE)) %as% sess,{
  
  sess$run(init)
  
  for (iteration in 1:num_train_iterations){
    
    batch <- next_batch(train_set, batch_size, num_time_steps)
    sess$run(train, feed_dict=dict(X = batch$X, y = batch$y))
    
    if (iteration %% 500 == 0){
      
      mse <- loss$eval(feed_dict=dict(X =batch$X,y=batch$y))
      cat(iteration, "\tMSE:", mse,'\n')
    }
  }
  saver$save(sess, "./mod_latest.ckpt")
})

#=============================================

with(tf$Session() %as% sess,{
  
  saver$restore(sess, "./mod_latest.ckpt")
  
  train_seed = train_set[(length(train_set)-num_time_steps+1):length(train_set)]
  
  for (iteration in 1:50){
    
    X_batch = array_reshape(train_seed[(length(train_seed)-num_time_steps+1):length(train_seed)], c(1, num_time_steps, 1))
    y_pred = sess$run(outputs, feed_dict= dict(X = X_batch))
    y_pred = array_reshape(y_pred, c(1,num_time_steps,1))
    train_seed <- c(train_seed, y_pred[1,num_time_steps,1])
  }
})

plot(train_seed[(num_time_steps+1):length(train_seed)], type = 'l', col= 'red', ylim = c(0,1))
lines(test_set)
