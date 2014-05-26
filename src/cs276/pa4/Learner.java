package cs276.pa4;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import util.Pair;
import util.PairComparator;
import weka.classifiers.Classifier;
import weka.core.Instances;

public abstract class Learner {
	
	/* Construct training features matrix */
	public abstract Instances extract_train_features(String train_data_file, String train_rel_file, Map<String,Double> idfs);

	/* Train the model */
	public abstract Classifier training (Instances dataset);
	
	/* Construct testing features matrix */
	public abstract TestFeatures extract_test_features(String test_data_file, Map<String,Double> idfs);
	
	/* Test the model, return ranked queries */
	public abstract Map<String, List<String>> testing(TestFeatures tf, Classifier model);
	
	/*
	 * Given a list of Pairs of (doc, score), return a sorted list of docs (by score)
	 */
	public List<String> getSortedDocs(List<Pair<String, Double>> docScores) {
		Collections.sort(docScores, new PairComparator());
		Collections.reverse(docScores);
		
		List<String> docs = new ArrayList<String>();
		for (Pair<String, Double> curr : docScores) {
			docs.add(curr.getFirst());
		}
		
		return docs;
	}
}
