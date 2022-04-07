library(ggplot2)
library(R.matlab)
library(tidyverse)

# 1. get the number of mice in each experiment 

mouse_name_zero_maze = list.files(path = "data/Zero_Maze/", pattern="*_*")

mouse_name_opp_sex = list.files(path = "data/Opp_Sex/", pattern="*_*")

mouse_name_dir_interact = list.files(path = "data/Dir_Interact/", pattern="*_*")

mouse_count_by_experiment = data.frame(c(length(mouse_name_dir_interact),length(mouse_name_opp_sex),length(mouse_name_zero_maze)),
                                       col.names = c("Dir_Interact","Opp_Sex","Zero_Maze"))

colnames(mouse_count_by_experiment) <- c("mouse_count","experiment")

ggplot(data = mouse_count_by_experiment, aes(x=experiment, y = mouse_count, fill=experiment)) +
  geom_bar(position = "dodge", stat = "identity") +
  geom_text(aes(label = mouse_count), vjust = 0, size = 4) +
  labs(title = "number of mice in each experiment")


ezm_409_behavior <- readMat("data/Zero_Maze/608034_409/Day_1/Trial_001_0/binned_behavior.mat") %>% 
  as.data.frame()

ezm_409_zscore <- readMat("data/Zero_Maze/608034_409/Day_1/Trial_001_0/binned_zscore.mat") %>% 
  as.data.frame()

ezm_412_zscore <- readMat("data/Zero_Maze/608102_412/Day_1/Trial_001_0/binned_zscore.mat") %>% 
  as.data.frame()

##-------------- a function to store the num of cells into df----

# create a function

cell_number <- function(exprmt = mouse_name_zero_maze){
  col_num = data.frame()
  
  for (i in 1:length(exprmt)){
    filename <- paste0("data/Zero_Maze/",exprmt[i],"/Day_1/Trial_001_0/binned_zscore.mat")
    name <- exprmt[i]
    data = readMat(filename) %>% as.data.frame()
    l <- cbind(ncol(data), name)
    col_num <- rbind(col_num, l)
    
  }
  col_num$V1 <- as.numeric(col_num$V1)
  
  return(col_num)
}


## 

# 2. get the number of cells for each mouse in each experiment

##-------------------- Zero maze ---------------------------
# read the number of cells for each data

cell_num_zero_maze <- cell_number()

# col_num = data.frame()
# 
# 
# for (i in 1:length(mouse_name_zero_maze)){
#   filename <- paste0("data/Zero_Maze/",mouse_name_zero_maze[i],"/Day_1/Trial_001_0/binned_zscore.mat")
#   name <- mouse_name_zero_maze[i]
#   data = readMat(filename) %>% as.data.frame()
#   l <- cbind(ncol(data), name)
#   col_num <- rbind(col_num, l)
#   # assign(x = name,value = data)
# }
# 
# col_num$V1 <- as.numeric(col_num$V1)



# plot the barplot

ggplot(data = cell_num_zero_maze, aes(x=name, y = V1, fill=name)) +
  geom_bar(position = "dodge", stat = "identity") +
  geom_text(aes(label = V1), vjust = 0, size = 4) +
  labs(title = "number of cells for each mouse in zero_maze")
  # geom_tile("Number of cells for each mouse in Zero_Maze")



##-------------------- mouse_name_opp_sex ---------------------------
# read the number of cells for each data

cell_num_opp_sex <- cell_number(mouse_name_opp_sex)



# plot the barplot

ggplot(data = cell_num_opp_sex, aes(x=name, y = V1, fill=name)) +
  geom_bar(position = "dodge", stat = "identity") +
  geom_text(aes(label = V1), vjust = 0, size = 4) +
  labs(title = "number of cells for each mouse in opp_sex")
  # geom_tile("Number of cells for each mouse in Zero_Maze")




test1 = cell_number(exprmt = mouse_name_opp_sex)

##-------------------- dir_interact ---------------------------
# read the number of cells for each data

cell_num_dir_interact <- cell_number(mouse_name_dir_interact)



# plot the barplot

ggplot(data = cell_num_dir_interact, aes(x=name, y = V1, fill=name)) +
  geom_bar(position = "dodge", stat = "identity") +
  geom_text(aes(label = V1), vjust = 0, size = 4) +
  labs(title = "number of cells for each mouse in dir_interact")
# geom_tile("Number of cells for each mouse in Zero_Maze")


# 3. get the 0,1 distribution by mouse by experiment 

## ----------------------zero_maze---------------------
col_num = data.frame()
for (i in 1:length(mouse_name_zero_maze)){
  filename <- paste0("data/Zero_Maze/",mouse_name_zero_maze[i],"/Day_1/Trial_001_0/binned_behavior.mat")
  name <- mouse_name_zero_maze[i]
  data = readMat(filename) %>% as.data.frame() %>% t() %>% as.data.frame() 
  data["name"] <- rep(name,nrow(data))
  col_num <- rbind(col_num, data)
  # assign(x = name,value = data)
}

col_num_stat <- col_num %>% group_by(name) %>% summarise_at(.vars = c(1,2), .funs = mean)

test1 <- table(col_num) %>% as.data.frame() %>% filter(Freq != 0)

for (i in 1:nrow(test1)){
  if (test1$V1[i] == 0 & test1$V2[i] ==0){
    test1$label[i] <-  "00"
  }
  if (test1$V1[i] == 0 & test1$V2[i] == 1){
    test1$label[i] <-  "01"
  }
  if (test1$V1[i] == 1 & test1$V2[i] ==0){
    test1$label[i] <-  "10"
  }
  
}

ggplot(data = test1, aes(x=label, y = Freq, fill=label)) +
  geom_bar(position = "dodge", stat = "identity") +
  geom_text(aes(label = Freq), vjust = 0, size = 4) +
  labs(title = "Get the behavior distribution by mouse in zero_maze") +
  facet_grid(~name)

## ----------------------opp_sex---------------------
col_num = data.frame()
for (i in 1:length(mouse_name_opp_sex)){
  filename <- paste0("data/Zero_Maze/",mouse_name_opp_sex[i],"/Day_1/Trial_001_0/binned_behavior.mat")
  name <- mouse_name_opp_sex[i]
  data = readMat(filename) %>% as.data.frame() %>% t() %>% as.data.frame() 
  data["name"] <- rep(name,nrow(data))
  col_num <- rbind(col_num, data)
  # assign(x = name,value = data)
}

col_num_stat <- col_num %>% group_by(name) %>% summarise_at(.vars = c(1,2), .funs = mean)

test2 <- table(col_num) %>% as.data.frame() %>% filter(Freq != 0)

for (i in 1:nrow(test2)){
  if (test2$V1[i] == 0 & test2$V2[i] ==0){
    test2$label[i] <-  "00"
  }
  if (test2$V1[i] == 0 & test2$V2[i] == 1){
    test2$label[i] <-  "01"
  }
  if (test2$V1[i] == 1 & test2$V2[i] ==0){
    test2$label[i] <-  "10"
  }
  
}

ggplot(data = test2, aes(x=label, y = Freq, fill=label)) +
  geom_bar(position = "dodge", stat = "identity") +
  geom_text(aes(label = Freq), vjust = 0, size = 4) +
  labs(title = "Get the behavior distribution by mouse in opp_sex") +
  facet_grid(~name)

## ----------------------dir_interact---------------------
col_num = data.frame()
for (i in 1:length(mouse_name_dir_interact)){
  filename <- paste0("data/Zero_Maze/",mouse_name_dir_interact[i],"/Day_1/Trial_001_0/binned_behavior.mat")
  name <- mouse_name_dir_interact[i]
  data = readMat(filename) %>% as.data.frame() %>% t() %>% as.data.frame() 
  data["name"] <- rep(name,nrow(data))
  col_num <- rbind(col_num, data)
  # assign(x = name,value = data)
}

col_num_stat <- col_num %>% group_by(name) %>% summarise_at(.vars = c(1,2), .funs = mean)

test3 <- table(col_num) %>% as.data.frame() %>% filter(Freq != 0)

for (i in 1:nrow(test3)){
  if (test3$V1[i] == 0 & test3$V2[i] ==0){
    test3$label[i] <-  "00"
  }
  if (test3$V1[i] == 0 & test3$V2[i] == 1){
    test3$label[i] <-  "01"
  }
  if (test3$V1[i] == 1 & test3$V2[i] ==0){
    test3$label[i] <-  "10"
  }
  
}

ggplot(data = test3, aes(x=label, y = Freq, fill=label)) +
  geom_bar(position = "dodge", stat = "identity") +
  geom_text(aes(label = Freq), vjust = 0, size = 4) +
  labs(title = "Get the behavior distribution by mouse in dir_interact") +
  facet_grid(~name)


