package tud.ke.ml.project.junit;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import org.junit.Test;

import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.keNN;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.SelectedTag;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.instance.RemovePercentage;

public class AdvancedValidation {

	public static void init(List<Instances> data) {
		ArffLoader loader = new ArffLoader();
		Instances instances = null;

		loader = new ArffLoader();
		try {
			loader.setFile(new File("data/credit-g.arff"));
			instances = loader.getDataSet();
			instances.setClassIndex(instances.numAttributes() - 1);
		}
		catch (IOException e) {
			e.printStackTrace();
		}

		data.add(instances);
	}

	RemovePercentage filterTrain;
	RemovePercentage filterTest;
	public static final int testSplitPercentage = 33;

	private void setUpSplittingFilter() {
		filterTrain = new RemovePercentage();
		filterTrain.setPercentage(testSplitPercentage);
		filterTest = new RemovePercentage();
		filterTest.setPercentage(testSplitPercentage);
		filterTest.setInvertSelection(true);
	}

	/**
	 * Compares the predictions of the implemented classifier and the reference Weka classifier
	 * 
	 * @param myClassifier classifier to be implemented by students
	 * @param wekaClassifier weka reference classifier
	 * @param testInstance the instance to classify
	 * @throws Exception
	 */
	public static void comparePredictions(keNN myClassifier, IBk wekaClassifier, Instance testInstance) throws Exception {
		double myClass = myClassifier.classifyInstance(testInstance);
		double[] wekaDistribution = wekaClassifier.distributionForInstance(testInstance);
		double maxProb = wekaDistribution[Utils.maxIndex(wekaDistribution)];
		assertTrue("Predicted class " + (int) myClass + " for [" + testInstance.toString() + "] not among top classes in predicted distribution by weka: " + Arrays.toString(wekaDistribution), wekaDistribution[(int) myClass] == maxProb);
	}

	/**
	 * This test validates if the model is getting learned without throwing exceptions.
	 * 
	 * @throws Exception
	 */
	@Test
	public void testLearnModel() throws Exception {
		keNN classifier = new keNN();
		List<Instances> data = new LinkedList<Instances>();

		init(data);

		for (Instances instances : data) {
			classifier.buildClassifier(instances);
		}
	}

	/**
	 * This test validates if the classifier is able to classify new instances without throwing exceptions.
	 * 
	 * @throws Exception
	 */
	@Test
	public void testClassify() throws Exception {
		keNN classifier = new keNN();
		List<Instances> data = new LinkedList<Instances>();

		setUpSplittingFilter();
		init(data);

		classifier.setkNearest(1);

		for (Instances instances : data) {
			filterTrain.setInputFormat(instances);
			filterTest.setInputFormat(instances);
			classifier.buildClassifier(Filter.useFilter(instances, filterTrain));
			for (Instance instance : Filter.useFilter(instances, filterTest)) {
				classifier.classifyInstance(instance);
			}
		}

		classifier.setkNearest(10);
		classifier.setMetric(new SelectedTag(1, keNN.TAGS_DISTANCE));
		classifier.setDistanceWeighting(new SelectedTag(1, keNN.TAGS_WEIGHTING));
		classifier.setNormalization(new SelectedTag(0, keNN.TAGS_NORM));

		for (Instances instances : data) {
			filterTrain.setInputFormat(instances);
			filterTest.setInputFormat(instances);
			classifier.buildClassifier(Filter.useFilter(instances, filterTrain));
			for (Instance instance : Filter.useFilter(instances, filterTest)) {
				classifier.classifyInstance(instance);
			}
		}
	}

	/**
	 * This test the correctness of the unweighted Manhattan distance implementation
	 * 
	 * @throws Exception
	 */
	@Test
	public void testCorrectnessUnweightedManhattank1() throws Exception {
		keNN classifier = new keNN();
		IBk wekaClassifier = new IBk();
		List<Instances> data = new LinkedList<Instances>();

		setUpSplittingFilter();

		NominalToBinary nomToBin = new NominalToBinary();

		setUpSplittingFilter();
		init(data);

		classifier.setkNearest(1);
		classifier.setMetric(new SelectedTag(0, keNN.TAGS_DISTANCE));
		classifier.setDistanceWeighting(new SelectedTag(0, keNN.TAGS_WEIGHTING));
		classifier.setNormalization(new SelectedTag(0, keNN.TAGS_NORM));

		wekaClassifier.setKNN(1);
		NearestNeighbourSearch search = new LinearNNSearch();
		ManhattanDistance df = new ManhattanDistance();
		df.setDontNormalize(true);
		search.setDistanceFunction(df);
		search.setMeasurePerformance(false);
		wekaClassifier.setNearestNeighbourSearchAlgorithm(search);
		wekaClassifier.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_NONE, IBk.TAGS_WEIGHTING));

		for (Instances instances : data) {
			nomToBin.setInputFormat(instances);
			instances = Filter.useFilter(instances, nomToBin);
			filterTrain.setInputFormat(instances);
			filterTest.setInputFormat(instances);
			Instances train = Filter.useFilter(instances, filterTrain);
			classifier.buildClassifier(train);
			wekaClassifier.buildClassifier(train);
			Instances test = Filter.useFilter(instances, filterTest);
			for (Instance instance : test) {
				comparePredictions(classifier, wekaClassifier, instance);
			}
		}
	}

	/**
	 * This test the correctness of the unweighted Manhattan distance implementation with nominal attributes
	 * 
	 * @throws Exception
	 */
	@Test
	public void testCorrectnessNominalUnweightedManhattank20() throws Exception {
		keNN classifier = new keNN();
		IBk wekaClassifier = new IBk();
		List<Instances> data = new LinkedList<Instances>();

		setUpSplittingFilter();

		init(data);

		classifier.setkNearest(20);
		classifier.setMetric(new SelectedTag(0, keNN.TAGS_DISTANCE));
		classifier.setDistanceWeighting(new SelectedTag(0, keNN.TAGS_WEIGHTING));
		classifier.setNormalization(new SelectedTag(0, keNN.TAGS_NORM));

		wekaClassifier.setKNN(20);
		NearestNeighbourSearch search = new LinearNNSearch();
		ManhattanDistance df = new ManhattanDistance();
		df.setDontNormalize(true);
		search.setDistanceFunction(df);
		search.setMeasurePerformance(false);
		wekaClassifier.setNearestNeighbourSearchAlgorithm(search);
		wekaClassifier.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_NONE, IBk.TAGS_WEIGHTING));

		for (Instances instances : data) {
			instances.setClassIndex(2);
			filterTrain.setInputFormat(instances);
			filterTest.setInputFormat(instances);
			Instances train = Filter.useFilter(instances, filterTrain);
			classifier.buildClassifier(train);
			wekaClassifier.buildClassifier(train);
			Instances test = Filter.useFilter(instances, filterTest);
			for (Instance instance : test) {
				comparePredictions(classifier, wekaClassifier, instance);
			}
		}
	}

	/**
	 * This test the correctness of the unweighted Manhattan distance implementation with nominal attributes
	 * 
	 * @throws Exception
	 */
	@Test
	public void testCorrectnessNominalWeightedManhattank10Normalized() throws Exception {
		keNN classifier = new keNN();
		IBk wekaClassifier = new IBk();
		List<Instances> data = new LinkedList<Instances>();

		setUpSplittingFilter();
		init(data);

		classifier.setkNearest(11);
		classifier.setMetric(new SelectedTag(0, keNN.TAGS_DISTANCE));
		classifier.setDistanceWeighting(new SelectedTag(1, keNN.TAGS_WEIGHTING));
		classifier.setNormalization(new SelectedTag(1, keNN.TAGS_NORM));

		wekaClassifier.setKNN(11);
		NearestNeighbourSearch search = new LinearNNSearch();
		ManhattanDistance df = new ManhattanDistance();
		df.setDontNormalize(false);
		search.setDistanceFunction(df);
		search.setMeasurePerformance(false);
		wekaClassifier.setNearestNeighbourSearchAlgorithm(search);
		wekaClassifier.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING));

		for (Instances instances : data) {
			instances.setClassIndex(2);
			filterTrain.setInputFormat(instances);
			filterTest.setInputFormat(instances);
			Instances train = Filter.useFilter(instances, filterTrain);
			classifier.buildClassifier(train);
			wekaClassifier.buildClassifier(train);
			Instances test = Filter.useFilter(instances, filterTest);
			for (Instance instance : test) {
				comparePredictions(classifier, wekaClassifier, instance);
			}
		}
	}

	/**
	 * This test validates the correctness of a higher k (11) classification
	 * 
	 * @throws Exception
	 */
	@Test
	public void testCorrectnessUnweightedEuclideank1() throws Exception {
		keNN classifier = new keNN();
		IBk wekaClassifier = new IBk();
		List<Instances> data = new LinkedList<Instances>();

		setUpSplittingFilter();
		init(data);

		classifier.setkNearest(11);
		classifier.setMetric(new SelectedTag(1, keNN.TAGS_DISTANCE));
		classifier.setDistanceWeighting(new SelectedTag(0, keNN.TAGS_WEIGHTING));
		classifier.setNormalization(new SelectedTag(0, keNN.TAGS_NORM));

		wekaClassifier.setKNN(11);
		NearestNeighbourSearch search = new LinearNNSearch();
		EuclideanDistance df = new EuclideanDistance();
		df.setDontNormalize(true);
		search.setDistanceFunction(df);
		search.setMeasurePerformance(false);
		wekaClassifier.setNearestNeighbourSearchAlgorithm(search);
		wekaClassifier.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_NONE, IBk.TAGS_WEIGHTING));

		for (Instances instances : data) {
			filterTrain.setInputFormat(instances);
			filterTest.setInputFormat(instances);
			Instances train = Filter.useFilter(instances, filterTrain);
			classifier.buildClassifier(train);
			wekaClassifier.buildClassifier(train);
			Instances test = Filter.useFilter(instances, filterTest);
			for (Instance instance : test) {
				comparePredictions(classifier, wekaClassifier, instance);
			}
		}
	}

	/**
	 * This test validates the correctness of a higher k (10) classification
	 * 
	 * @throws Exception
	 */
	@Test
	public void testCorrectnessWeightedManhattank1() throws Exception {
		keNN classifier = new keNN();
		IBk wekaClassifier = new IBk();
		List<Instances> data = new LinkedList<Instances>();

		NominalToBinary nomToBin = new NominalToBinary();

		setUpSplittingFilter();
		init(data);

		classifier.setkNearest(1);
		classifier.setMetric(new SelectedTag(0, keNN.TAGS_DISTANCE));
		classifier.setDistanceWeighting(new SelectedTag(1, keNN.TAGS_WEIGHTING));
		classifier.setNormalization(new SelectedTag(0, keNN.TAGS_NORM));

		wekaClassifier.setKNN(1);
		NearestNeighbourSearch search = new LinearNNSearch();
		ManhattanDistance df = new ManhattanDistance();
		df.setDontNormalize(true);
		search.setDistanceFunction(df);
		search.setMeasurePerformance(false);
		wekaClassifier.setNearestNeighbourSearchAlgorithm(search);
		wekaClassifier.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING));

		for (Instances instances : data) {
			nomToBin.setInputFormat(instances);
			instances = Filter.useFilter(instances, nomToBin);
			filterTrain.setInputFormat(instances);
			filterTest.setInputFormat(instances);
			Instances train = Filter.useFilter(instances, filterTrain);
			classifier.buildClassifier(train);
			wekaClassifier.buildClassifier(train);
			Instances test = Filter.useFilter(instances, filterTest);
			for (Instance instance : test) {
				comparePredictions(classifier, wekaClassifier, instance);
			}
		}
	}

	/**
	 * This tests validates the inverse weighted, Euclidean distance metric.
	 * 
	 * @throws Exception
	 */
	@Test
	public void testCorrectnessWeightedEuclideank1() throws Exception {
		keNN classifier = new keNN();
		IBk wekaClassifier = new IBk();
		List<Instances> data = new LinkedList<Instances>();

		setUpSplittingFilter();
		init(data);

		NearestNeighbourSearch search = new LinearNNSearch();

		classifier.setkNearest(1);
		classifier.setMetric(new SelectedTag(1, keNN.TAGS_DISTANCE));
		classifier.setDistanceWeighting(new SelectedTag(1, keNN.TAGS_WEIGHTING));
		classifier.setNormalization(new SelectedTag(0, keNN.TAGS_NORM));

		wekaClassifier.setKNN(1);
		search = new LinearNNSearch();
		EuclideanDistance df = new EuclideanDistance();
		df.setDontNormalize(true);
		search.setDistanceFunction(df);
		search.setMeasurePerformance(false);
		wekaClassifier.setNearestNeighbourSearchAlgorithm(search);
		wekaClassifier.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING));

		for (Instances instances : data) {
			filterTrain.setInputFormat(instances);
			filterTest.setInputFormat(instances);
			Instances train = Filter.useFilter(instances, filterTrain);
			classifier.buildClassifier(train);
			wekaClassifier.buildClassifier(train);
			Instances test = Filter.useFilter(instances, filterTest);
			for (Instance instance : test) {
				comparePredictions(classifier, wekaClassifier, instance);
			}
		}
	}

	/**
	 * This tests validates the inverse weighted, euclidean distance metric.
	 * 
	 * @throws Exception
	 */
	@Test
	public void testCorrectnessWeightedEuclideank1normalized() throws Exception {
		keNN classifier = new keNN();
		IBk wekaClassifier = new IBk();
		List<Instances> data = new LinkedList<Instances>();

		setUpSplittingFilter();

		NominalToBinary nomToBin = new NominalToBinary();

		init(data);

		NearestNeighbourSearch search = new LinearNNSearch();

		classifier.setkNearest(1);
		classifier.setMetric(new SelectedTag(1, keNN.TAGS_DISTANCE));
		classifier.setDistanceWeighting(new SelectedTag(1, keNN.TAGS_WEIGHTING));
		classifier.setNormalization(new SelectedTag(1, keNN.TAGS_NORM));

		wekaClassifier.setKNN(1);
		search = new LinearNNSearch();
		EuclideanDistance df = new EuclideanDistance();
		df.setDontNormalize(false);
		search.setDistanceFunction(df);
		search.setMeasurePerformance(false);
		wekaClassifier.setNearestNeighbourSearchAlgorithm(search);
		wekaClassifier.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING));

		for (Instances instances : data) {
			nomToBin.setInputFormat(instances);
			instances = Filter.useFilter(instances, nomToBin);
			filterTrain.setInputFormat(instances);
			filterTest.setInputFormat(instances);
			Instances train = Filter.useFilter(instances, filterTrain);
			classifier.buildClassifier(train);
			wekaClassifier.buildClassifier(train);
			Instances test = Filter.useFilter(instances, filterTest);
			for (Instance instance : test) {
				comparePredictions(classifier, wekaClassifier, instance);
			}
		}
	}
}
