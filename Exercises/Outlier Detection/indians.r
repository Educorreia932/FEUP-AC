library(performanceEstimation)
library(mlbench)
library(Rcpp)
library(e1071)
library(AUC)
library(solitude)

data("PimaIndiansDiabetes")

dataset <- PimaIndiansDiabetes
formula <- diabetes ~ .

# Function to calculate AUC
AUC <- function(trues, preds, ...) {
	c(auc = AUC::auc(roc(preds, trues)))
}

# 1.1 Isolation

od.if <- function(form, train, test, ntrees = 100, ...) {
	tgt <- which(colnames(train) == as.character(form[[2]]))
	iso <- isolationForest$new(num_trees = ntrees)
	
	iso$fit(train, ...)
	
	p <- iso$predict(test)
	p <- p$anomaly_score
	p <- scale(p)
	
	res <- list(trues = test[, tgt], preds = p)
	
	res
}


# 1.2 One-class SVM Linear

od.svm.linear <- function(form, train, test, classMaj, n = 0.1) {
	tgt <- which(colnames(train) == as.character(form[[2]]))
	j <- which(train[, tgt] == classMaj)
	m.svm <-
		svm(form,
			train[j, ],
			type = "one-classification",
			kernel = "linear",
			nu = n)
	p <- predict(m.svm, test)
	res <- list(trues = test[, tgt], preds = p)
	
	res
}

exp.od.svm.linear <- performanceEstimation::performanceEstimation(
	PredTask(formula, dataset),
	c(workflowVariants(
		"od.svm.linear", classMaj = 'neg',
		n = c(seq(0.1, 0.9, by = 0.1))
	)),
	EstimationTask(
		metrics = "auc",
		method = CV(nReps = 1, nFolds = 10),
		evaluator = "AUC"
	)
)


plot(exp.od.svm.linear)
rankWorkflows(exp.od.svm.linear, maxs = TRUE)
getWorkflow("od.svm.linear.v3", exp.od.svm.linear)