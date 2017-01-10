package tud.ke.ml.project.classifier;

import java.io.Serializable;
import java.util.*;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
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
            count = count + (1/nearestN.getB());
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
            if ( votes.get(okey) == max )
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
    protected List<Pair<List<Object>, Double>> getNearest(List<Object> data) {
        ArrayList<Pair<List<Object>, Double>> distances = new ArrayList<>();
        for (List<Object> instance : this.model) {
            if(this.getMetric() == 0)
                distances.add(new Pair<>(instance, this.determineManhattanDistance(instance, data)));
            else
                distances.add(new Pair<>(instance, this.determineEuclideanDistance(instance, data)));
        }
        distances.sort((x, y) -> (int) (x.getB() * 10 - y.getB() * 10));
        return distances.subList(0, super.getkNearest());
    }

    @Override
    protected double determineManhattanDistance(List<Object> instance1, List<Object> instance2) {
        throw new NotImplementedException();
    }

    @Override
    protected double determineEuclideanDistance(List<Object> instance1, List<Object> instance2) {
        throw new NotImplementedException();
    }

    @Override
    protected double[][] normalizationScaling() {
        throw new NotImplementedException();
    }

}
