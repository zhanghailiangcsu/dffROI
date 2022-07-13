library(KPIC)

kpic_pic <- function(filename,level = 1000){
  raw <- LoadData(filename)
  pics <- getPIC.kmeans(raw, level=level)
  pics <- PICsplit(pics)
  pics <- getPeaks(pics)
  return (pics)
}

kpic_pic_set <- function(filename,level = 1000){
  PICS <- PICset.kmeans(filename, level=level, export=F, par=T)
  PICS <- PICset.split(PICS)
  return(PICS)
}

kpic_pic_getpeak <- function(PICS){
  PICS <- PICset.getPeaks(PICS)
  return(PICS)
}


kpic_group <- function(PICS,tolerance = c(0.1, 20)){
  groups_raw <- PICset.group(PICS, tolerance = tolerance)
  groups_align <- PICset.align(groups_raw, method='fftcc',move='loess')
  groups_align <- PICset.group(groups_align$picset, tolerance = tolerance)
  groups_align <- PICset.align(groups_align, method='fftcc',move='direct')
  return(groups_align)
}

kpic_iso <-function(groups_align){
  groups_align <- groupCombine(groups_align, type='isotope')
  return(groups_align)
}

kpic_mat <- function(groups_align){
  data <- getDataMatrix(groups_align)
  return(data)
}

kpic_fill <- function(data){
  data <- fillPeaks.EIBPC(data)
  return(data)
}

kpic_select <- function(a,pic){
  for ( i in a){
    pic[["pics"]][i] <- NULL
    pic[["peaks"]][i] <- NULL
  }
  return(pic)
}

kpic_pattern <- function(data){
  labels <- c(rep('A',6), rep('B',6))
  analyst.RF(labels, data$data.mat)
}

















