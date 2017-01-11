package tud.ke.ml.project.classifier;

import java.io.Serializable;
import java.util.*;

import tud.ke.ml.project.util.Pair;

/**
 * This implementation assumes the class attribute is always available (but probably not set).
 */
public class NearestNeighbor extends INearestNeighbor implements Serializable {
    private static final long serialVersionUID = 1L;

    private List<List<Object>> model;

    protected double[] scaling;
    protected double[] translation;

    @Override
    public String getMatrikelNumbers() {
        return "2879718,2594213,2753711";
    }

    @Override
    protected void learnModel(List<List<Object>> data) {
        this.model = data;

    }

    @Override
    protected Map<Object, Double> getUnweightedVotes(List<Pair<List<Object>, Double>> subset) {
        Map<Object, Double> unweightedVotes = new HashMap<>();
        //initialize unweightedVotes with all possible class attributes
        for (Pair<List<Object>, Double> nearestN : subset) {
            unweightedVotes.put(nearestN.getA().get(this.getClassAttribute()), 0.0D);
        }
        //Count the number of each class attribute in the nearestN
        for (Pair<List<Object>, Double> nearestN : subset) {
            double count = unweightedVotes.get(nearestN.getA().get(this.getClassAttribute()));
            count++;
            unweightedVotes.put(nearestN.getA().get(this.getClassAttribute()), count);
        }
        return unweightedVotes;
    }

    @Override
    protected Map<Object, Double> getWeightedVotes(List<Pair<List<Object>, Double>> subset) {
        Map<Object, Double> weightedVotes = new HashMap<>();
        //initialize weightedVotes with all possible class attributes
        for (Pair<List<Object>, Double> nearestN : subset) {
            weightedVotes.put(nearestN.getA().get(this.getClassAttribute()), 0.0D);
        }
        //Count the number of each class attribute in the nearestN
        for (Pair<List<Object>, Double> nearestN : subset) {
            double count = weightedVotes.get(nearestN.getA().get(this.getClassAttribute()));
            //Votes are the sum of the inverted distances
            count = count + (1 / nearestN.getB());
            weightedVotes.put(nearestN.getA().get(this.getClassAttribute()), count);
        }
        return weightedVotes;
    }

    @Override
    protected Object getWinner(Map<Object, Double> votes) {
        double max = Double.MIN_VALUE;
        for (Double d : votes.values()) {
            if (max < d)
                max = d;
        }
        for (Object okey : votes.keySet()) {
            if (votes.get(okey) == max)
                return okey;
        }
        return null;
    }

    @Override
    protected Object vote(List<Pair<List<Object>, Double>> subset) {
        return this.isInverseWeighting() ? this.getWinner(this.getWeightedVotes(subset)) :
                this.getWinner(this.getUnweightedVotes(subset));
    }

    @Override
    protected List<Pair<List<Object>, Double>> getNearest(List<Object> originalInput) {
        ArrayList<Pair<List<Object>, Double>> distances = new ArrayList<>();

        List<Object> workingInput = this.copyData(originalInput);

        if (isNormalizing()) {
            double[][] translationScaling = this.normalizationScaling();
            translation = translationScaling[0];
            scaling = translationScaling[1];
            scaleTranslate(workingInput);
        }

        List<List<Object>> data = this.deepCopyData(this.model);

        for (List<Object> instance : data) {
            if (isNormalizing()) {
                scaleTranslate(instance);
            }

            if (this.getMetric() == 0)
                distances.add(new Pair<>(instance, this.determineManhattanDistance(instance, workingInput)));
            else
                distances.add(new Pair<>(instance, this.determineEuclideanDistance(instance, workingInput)));
        }
        distances.sort((x, y) -> (int) (x.getB() * 1000000000000L - y.getB() * 1000000000000L));
        return distances.subList(0, super.getkNearest());
    }

    private List<List<Object>> deepCopyData(List<List<Object>> toCopy) {
        List<List<Object>> data = new ArrayList<>();
        for (List<Object> innerlist : toCopy) {
            data.add(this.copyData(innerlist));
        }
        return data;
    }

    private List<Object> copyData(List<Object> toCopy) {
        List<Object> data = new ArrayList<>();
        for (Object o : toCopy) {
            if (o instanceof String) {
                data.add(new String((String) o));
            } else if (o instanceof Double) {
                data.add(new Double((Double) o));
            }
        }
        return data;
    }

    private void scaleTranslate(List<Object> instance) {
        for (int i = 0; i < instance.size(); i++) {
            Object o = instance.get(i);
            if (o instanceof Double) {
                Double d = (Double) o;
                d -= translation[i];
                d /= scaling[i];
                instance.set(i, d);
            }
        }
    }

    private List<Double> getPlainDistances(List<Object> instance1, List<Object> instance2) {
        List<Double> plainDistances = new ArrayList<>();

        for (int i = 0; i < instance1.size(); i++) {
            if (i == this.getClassAttribute())
                continue;

            Object attr1 = instance1.get(i);
            Object attr2 = instance2.get(i);

            if (attr1 instanceof String) {
                plainDistances.add(attr1.equals(attr2) ? 0d : 1d);
            } else if (attr1 instanceof Double) {
                plainDistances.add(Math.abs((Double) attr1 - (Double) attr2));
            }
        }

        return plainDistances;
    }

    @Override
    protected double determineManhattanDistance(List<Object> instance1, List<Object> instance2) {
        List<Double> plainDinstances = this.getPlainDistances(instance1, instance2);
        return plainDinstances.stream().mapToDouble(d -> d).sum();
    }

    @Override
    protected double determineEuclideanDistance(List<Object> instance1, List<Object> instance2) {
        List<Double> plainDinstances = this.getPlainDistances(instance1, instance2);
        return Math.sqrt(plainDinstances.stream().mapToDouble(d -> d * d).sum());
    }

    @Override
    protected double[][] normalizationScaling() {
        double[][] translationScalation = new double[2][this.model.get(1).size()];
        double[][] minMax = new double[2][translationScalation[0].length];

        for (int i = 0; i < minMax[0].length; i++) {
            minMax[0][i] = Double.MAX_VALUE;
            minMax[1][i] = -Double.MAX_VALUE;
        }

        for (List<Object> innerList : model) {
            int index = 0;
            for (Object o : innerList) {
                if (o instanceof Double) {
                    double min = minMax[0][index];
                    double max = minMax[1][index];
                    Double d = (Double) o;
                    if (d < min)
                        minMax[0][index] = d;
                    if (d > max)
                        minMax[1][index] = d;
                }
                index++;
            }
        }

        translationScalation[0] = minMax[0];
        for (int i = 0; i < minMax[0].length; i++) {
            translationScalation[1][i] = minMax[1][i] - minMax[0][i];
            if (translationScalation[1][i] == 0d) {
                translationScalation[1][i] = 1d;
            }
        }
        return translationScalation;
    }

}
