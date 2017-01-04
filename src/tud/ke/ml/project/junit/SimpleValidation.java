package tud.ke.ml.project.junit;

import static org.junit.Assert.assertNotNull;

import java.io.File;
import java.util.LinkedList;
import java.util.List;

import org.junit.BeforeClass;
import org.junit.Test;

import tud.ke.ml.project.classifier.NearestNeighbor;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.keNN;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.SelectedTag;
import weka.core.converters.ArffLoader;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class SimpleValidation {

	private static List<Instances> data;
	private static RemovePercentage filterTrain, filterTest;
	private static keNN classifier = new keNN();
	private static IBk wekaClassifier = new IBk();

	/**
	 * This test validates if the matrikel numbers are returned.
	 * 
	 * @throws Exception
	 */
	@Test
	public void testGroupNumber() throws Exception {
		NearestNeighbor classifier = new NearestNeighbor();
		assertNotNull(classifier.getMatrikelNumbers());
	}

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		data = new LinkedList<Instances>();
		ArffLoader loader = new ArffLoader();
		Instances instances;

		loader.setFile(new File("data/contact-lenses.arff"));
		instances = loader.getDataSet();
		instances.setClassIndex(instances.numAttributes() - 1);
		data.add(instances);

		classifier = new keNN();
		wekaClassifier = new IBk();

		filterTrain = new RemovePercentage();
		filterTrain.setPercentage(AdvancedValidation.testSplitPercentage);
		filterTest = new RemovePercentage();
		filterTest.setPercentage(AdvancedValidation.testSplitPercentage);
		filterTest.setInvertSelection(true);
	}

	/**
	 * This test validates if the model is getting learned without throwing exceptions.
	 * 
	 * @throws Exception
	 */
	@Test
	public void testLearnModel() throws Exception {
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
		classifier.setkNearest(2);

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
		classifier.setkNearest(1);
		classifier.setMetric(new SelectedTag(0, keNN.TAGS_DISTANCE));
		classifier.setDistanceWeighting(new SelectedTag(0, keNN.TAGS_WEIGHTING));

		wekaClassifier.setKNN(1);
		NearestNeighbourSearch search = new LinearNNSearch();
		ManhattanDistance df = new ManhattanDistance();
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
				AdvancedValidation.comparePredictions(classifier, wekaClassifier, instance);
			}
		}
	}

	/**
	 * This test validates the correctness of a higher k (10) classification
	 * 
	 * @throws Exception
	 */
	@Test
	public void testCorrectnessUnweightedEuclideank1() throws Exception {
		classifier.setkNearest(1);
		classifier.setMetric(new SelectedTag(1, keNN.TAGS_DISTANCE));
		classifier.setDistanceWeighting(new SelectedTag(0, keNN.TAGS_WEIGHTING));

		wekaClassifier.setKNN(1);
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
				AdvancedValidation.comparePredictions(classifier, wekaClassifier, instance);
			}
		}
	}

	/**
	 * This test validates the correctness of a higher k (10) classification
	 * 
	 * @throws Exception
	 */
	@Test
	public void testCorrectnessWeightedManhattank10() throws Exception {
		classifier.setkNearest(10);
		classifier.setMetric(new SelectedTag(0, keNN.TAGS_DISTANCE));
		classifier.setDistanceWeighting(new SelectedTag(1, keNN.TAGS_WEIGHTING));

		wekaClassifier.setKNN(10);
		NearestNeighbourSearch search = new LinearNNSearch();
		ManhattanDistance df = new ManhattanDistance();
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
				AdvancedValidation.comparePredictions(classifier, wekaClassifier, instance);
			}
		}
	}

	/**
	 * This tests validates the inverse weighted, euclidean distance metric.
	 * 
	 * @throws Exception
	 */
	@Test
	public void testCorrectnessWeightedEuclideank10() throws Exception {
		NearestNeighbourSearch search = new LinearNNSearch();

		classifier.setkNearest(10);
		classifier.setMetric(new SelectedTag(1, keNN.TAGS_DISTANCE));
		classifier.setDistanceWeighting(new SelectedTag(1, keNN.TAGS_WEIGHTING));

		wekaClassifier.setKNN(10);
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
				AdvancedValidation.comparePredictions(classifier, wekaClassifier, instance);
			}
		}
	}
}
