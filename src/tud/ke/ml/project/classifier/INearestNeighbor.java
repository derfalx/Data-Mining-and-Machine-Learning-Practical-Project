package tud.ke.ml.project.classifier;

import java.util.List;
import java.util.Map;

import tud.ke.ml.project.util.Pair;

/**
 * Abstract base class for the k-NearestNeighbour classifier.
 * 
 * DO NOT MODIFY
 * 
 */
public abstract class INearestNeighbor {

	private int classAttribute;

	private int kNearest = 1, metric;
	private boolean inverseWeighting, normalize;

	/**
	 * DO NOT CALL.
	 * 
	 * @param i
	 */
	protected void setClassAttribute(int i) {
		classAttribute = i;
	}

	/**
	 * @return the index of the class attribute
	 */
	protected int getClassAttribute() {
		return classAttribute;
	}

	/**
	 * @return the amount of nearest neighbors to use for voting (k)
	 */
	public int getkNearest() {
		return kNearest;
	}

	/**
	 * DO NOT CALL.
	 * 
	 * @param kNearest the amount of nearest neighbors to use for voting (k)
	 */
	public void setkNearest(int kNearest) {
		this.kNearest = kNearest;
	}

	/**
	 * Returns the distance metric to use
	 * 
	 * @return 0 for Manhattan, 1 for Euclidean
	 */
	public int getMetric() {
		return metric;
	}

	/**
	 * DO NOT CALL.
	 * 
	 * @param the distance metric to use
	 */
	public void setMetric(int metric) {
		this.metric = metric;
	}

	/**
	 * Determines if inverse distance weighting or unweighed voting should be use for the voting
	 * 
	 * @return true if inverse distance weighting is to be used
	 */
	public boolean isInverseWeighting() {
		return inverseWeighting;
	}

	/**
	 * DO NOT CALL.
	 * 
	 * @param true if inverse distance weighting is to be used
	 */
	public void setInverseWeighting(boolean inverseWeighting) {
		this.inverseWeighting = inverseWeighting;
	}

	/**
	 * DO NOT CALL.
	 * 
	 * @param b
	 */
	public void setNormalizing(boolean b) {
		normalize = b;
	}

	/**
	 * True if normalization is to be used
	 * 
	 * @return
	 */
	public boolean isNormalizing() {
		return normalize;
	}

	/**
	 * DO NOT CALL.
	 * 
	 * @param data
	 * @param classAttribute
	 */
	public void learnModel(List<List<Object>> traindata, int classAttribute) {
		this.classAttribute = classAttribute;
		learnModel(traindata);
	}

	/**
	 * DO NOT CALL.
	 * 
	 * @param data
	 * @param classAttribute
	 * @return
	 */
	public Object classifyInstance(List<Object> testdata, int classAttribute) {
		List<Pair<List<Object>, Double>> subset = getNearest(testdata);
		Object classValue = vote(subset);
		return classValue;
	}

	/**
	 * Determines the winning class base on the subset of nearest neighbors
	 * 
	 * @param subset Set of nearest neighbors with their distance
	 * @return the winning class, ususally a String
	 */
	protected abstract Object vote(List<Pair<List<Object>, Double>> subset);

	/**
	 * Learns the model
	 * 
	 * @param data the training data
	 */
	protected abstract void learnModel(List<List<Object>> traindata);

	/**
	 * Collects the votes based on an unweighted schema
	 * 
	 * @param subset Set of nearest neighbors with their distance
	 * @return Map of classes with their votes (e.g. returnValue.get("yes") are the votes for class "yes")
	 */
	protected abstract Map<Object, Double> getUnweightedVotes(List<Pair<List<Object>, Double>> subset);

	/**
	 * Collects the votes based on the inverse distance weighting schema
	 * 
	 * @param subset Set of nearest neighbors with their distance
	 * @return Map of classes with their votes (e.g. returnValue.get("yes") are the votes for class "yes")
	 */
	protected abstract Map<Object, Double> getWeightedVotes(List<Pair<List<Object>, Double>> subset);

	/**
	 * Determines the winning class based on the given votes
	 * 
	 * @param votesFor List of classes with their votes (e.g. returnValue.get("yes") are the votes for class "yes")
	 * @return the winning class, usually a String
	 */
	protected abstract Object getWinner(Map<Object, Double> votesFor);

	/**
	 * Calculates the nearest neighbors. Must call {@link #determineManhattanDistance(List, List)} or {@link #determineEuclideanDistance(List, List)} according to {@link #getMetric()}
	 * 
	 * @param data the current test instance
	 * @return a list of {@link #getkNearest()} nearest instances with their according distance
	 */
	protected abstract List<Pair<List<Object>, Double>> getNearest(List<Object> testdata);

	/**
	 * Calculates the Manhattan distance between the two instances
	 * 
	 * @param instance1
	 * @param instance2
	 * @return the distance
	 */
	protected abstract double determineManhattanDistance(List<Object> instance1, List<Object> instance2);

	/**
	 * Calculates the Euclidean distance between the two instances
	 * 
	 * @param instance1
	 * @param instance2
	 * @return the distance
	 */
	protected abstract double determineEuclideanDistance(List<Object> instance1, List<Object> instance2);

	/**
	 * Calculates the scaling and translation factor for each attribute
	 * 
	 * @return an array of all scaling factors and translation factors: [[scaling],[translation]]
	 */
	protected abstract double[][] normalizationScaling();

	/**
	 * 
	 * @return the matrikel numbers of students in the group
	 */
	protected abstract String getMatrikelNumbers();

}
