package weka.classifiers.lazy;

import java.util.Collections;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;

import tud.ke.ml.project.classifier.NearestNeighbor;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.Utils;

/**
 * WEKA Wrapper class. Do not modify.
 * 
 */
public class keNN extends AbstractClassifier implements OptionHandler {

	private static final long serialVersionUID = 923612147248506865L;

	public static final int WEIGHT_NONE = 0;
	public static final int WEIGHT_INVERSE = 1;
	public static final Tag[] TAGS_WEIGHTING = { new Tag(WEIGHT_NONE, "No distance weighting"), new Tag(WEIGHT_INVERSE, "Weight by 1/distance"), };

	public static final int DIST_MANHATTAN = 0;
	public static final int DIST_EUCLIDEAN = 1;
	public static final Tag[] TAGS_DISTANCE = { new Tag(DIST_MANHATTAN, "Manhattan distance"), new Tag(DIST_EUCLIDEAN, "Euclidean distance"), };

	public static final int NORM_TRUE = 0;
	public static final int NORM_FALSE = 1;
	public static final Tag[] TAGS_NORM = { new Tag(NORM_FALSE, "No normalization"), new Tag(NORM_TRUE, "Normalize variables"), };

	private NearestNeighbor classifier = new NearestNeighbor();

	private boolean[] isNumeric;

	@Override
	public void buildClassifier(Instances instances) throws Exception {
		List<List<Object>> data = new LinkedList<List<Object>>();
		int classAttribute = instances.classAttribute().index();
		determineNumericAttributes(instances);
		for (Instance inst : instances) {
			data.add(convert(inst));
		}
		classifier.learnModel(data, classAttribute);
	}

	@Override
	public double classifyInstance(Instance instance) {
		int classAttribute = instance.classAttribute().index();
		Object classValue = classifier.classifyInstance(convert(instance), classAttribute);
		return instance.classAttribute().indexOfValue(classValue.toString());
	}

	public String globalInfo() {
		return "KE Project Nearest Neighbour Classifier";
	}

	private void determineNumericAttributes(Instances instances) {
		isNumeric = new boolean[instances.numAttributes()];
		for (int i = 0; i < isNumeric.length; i++) {
			if (instances.attribute(i).isNumeric())
				isNumeric[i] = true;
			else
				isNumeric[i] = false;
		}
	}

	private List<Object> convert(Instance instance) {
		List<Object> data = new LinkedList<Object>();
		for (int i = 0; i < isNumeric.length; i++) {
			if (isNumeric[i]) {
				data.add(instance.value(i));
			}
			else {
				data.add(instance.attribute(i).value((int) instance.value(i)));
			}
		}
		return data;
	}

	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>(7);

		newVector.addElement(new Option("\tNumber of nearest neighbours (k) used in classification.\n" + "\t(Default = 1)", "K", 1, "-K <number of neighbors>"));
		newVector.addElement(new Option("\tUse weighted voting by the inverse of their distance\n" + "\t(use when k > 1)", "I", 0, "-I"));
		newVector.addElement(new Option("\tUse euclidean distance instead of manhattan.\n", "E", 0, "-E "));
		newVector.addAll(Collections.list(super.listOptions()));

		return newVector.elements();
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		String knnString = Utils.getOption('K', options);

		if (knnString.length() != 0) {
			classifier.setkNearest(Integer.parseInt(knnString));
		}
		else {
			classifier.setkNearest(1);
		}

		if (Utils.getFlag('I', options)) {
			classifier.setInverseWeighting(true);
		}
		else {
			classifier.setInverseWeighting(false);
		}

		if (Utils.getFlag('E', options)) {
			classifier.setMetric(1);
		}
		else {
			classifier.setMetric(0);
		}
	}

	@Override
	public String[] getOptions() {
		Vector<String> options = new Vector<String>();
		options.add("-K");
		options.add("" + classifier.getkNearest());

		if (classifier.isInverseWeighting()) {
			options.add("-I");
		}

		if (classifier.getMetric() == 1) {
			options.add("-E");
		}

		Collections.addAll(options, super.getOptions());

		return options.toArray(new String[0]);
	}

	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);

		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NUMERIC_CLASS);

		// instances
		result.setMinimumNumberInstances(1);

		return result;
	}

	public void setkNearest(int kNearest) {
		classifier.setkNearest(kNearest);
	}

	public int getkNearest() {
		return classifier.getkNearest();
	}

	public String kNearestTipText() {
		return "Amount of nearest neighbours to use (k)";
	}

	public String distanceWeightingTipText() {

		return "The distance weighting method used.";
	}

	public SelectedTag getDistanceWeighting() {

		return new SelectedTag(classifier.isInverseWeighting() ? 1 : 0, TAGS_WEIGHTING);
	}

	public void setDistanceWeighting(SelectedTag newMethod) {

		if (newMethod.getTags() == TAGS_WEIGHTING) {
			classifier.setInverseWeighting(newMethod.getSelectedTag().getID() == 0 ? false : true);
		}
	}

	public String metricTipText() {

		return "The distance metric used.";
	}

	public SelectedTag getMetric() {

		return new SelectedTag(classifier.getMetric(), TAGS_DISTANCE);
	}

	public void setMetric(SelectedTag newMethod) {

		if (newMethod.getTags() == TAGS_DISTANCE) {
			classifier.setMetric(newMethod.getSelectedTag().getID());
		}
	}

	public SelectedTag getNormalization() {

		return new SelectedTag(classifier.isNormalizing() ? 1 : 0, TAGS_NORM);
	}

	public void setNormalization(SelectedTag newMethod) {

		if (newMethod.getTags() == TAGS_NORM) {
			classifier.setNormalizing(newMethod.getSelectedTag().getID() == 0 ? false : true);
		}
	}
}
