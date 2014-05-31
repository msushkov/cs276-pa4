package cs276.pa4;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import util.Pair;
import util.PairComparator;
import weka.classifiers.Classifier;
import weka.core.Instances;

public abstract class Learner {
	
	/* Construct training features matrix */
	public abstract Instances extract_train_features(String train_data_file, String train_rel_file,
			Map<String,Double> idfs, Map<String, Double> additionalFeatures);

	/* Train the model */
	public abstract Classifier training (Instances dataset);
	
	/* Construct testing features matrix */
	public abstract TestFeatures extract_test_features(String test_data_file,
			Map<String,Double> idfs, Map<String, Double> additionalFeatures);
	
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
	
	/*
	 * Sorts the list of documents based on the pairwise comparisons.
	 */
	public List<String> getSortedDocsPairwise(List<String> docs, final Map<Pair<String, String>, Double> pairWiseScores) {
		Collections.sort(docs, new Comparator() {

			@Override
			public int compare(Object doc1, Object doc2) {
				Pair<String, String> docPair = new Pair((String) doc1, (String) doc2);
		
				double val = pairWiseScores.get(docPair);
				
				if (val == 0.0) {
					return -1;
				} else {
					return 1;
				}
			}
			
		});
		Collections.reverse(docs);

		return docs;
	}
	
	public void debugPrintVector(double[] vec) {
		for (int i = 0; i < vec.length; i++) {
			System.out.print(vec[i] + ", ");
		}
		System.out.println();
	}
}
