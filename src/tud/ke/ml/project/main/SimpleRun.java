package tud.ke.ml.project.main;

import java.io.File;

import weka.classifiers.lazy.keNN;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class SimpleRun {
	private static Instances data;
	private static RemovePercentage filterTrain, filterTest;

	public static void main(String[] args) throws Exception {
		setUpData();
		runClassifier();
	}

	public static void setUpData() throws Exception {
		// load data:
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File("data/contact-lenses.arff"));
		data = loader.getDataSet();

		// set class attribute:
		data.setClassIndex(data.numAttributes() - 1);

		// create training and test splits:
		double percentageSplit = 60;
		filterTrain = new RemovePercentage();
		filterTrain.setPercentage(percentageSplit);
		filterTrain.setInvertSelection(true);

		filterTest = new RemovePercentage();
		filterTest.setPercentage(percentageSplit);
		filterTrain.setInputFormat(data);
		filterTest.setInputFormat(data);
	}

	public static void runClassifier() throws Exception {
		// create classifier:
		keNN classifier = new keNN();

		// build model:
		classifier.buildClassifier(Filter.useFilter(data, filterTrain));

		// classify test instances:
		Instances testData = Filter.useFilter(data, filterTest);
		for (Instance instance : testData) {
			double result = classifier.classifyInstance(instance);
			String klasse = "" + result;
			if (instance.classAttribute().isNominal())
				klasse = instance.classAttribute().value((int) result);
			System.out.println("Die Instanz \"" + instance.toString() + "\" wurde klassifiziert als Klasse \"" + klasse + "\"");
		}
	}

}
