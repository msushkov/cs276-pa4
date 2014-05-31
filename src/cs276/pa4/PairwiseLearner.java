package cs276.pa4;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

import util.Pair;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Standardize;

public class PairwiseLearner extends Learner {

	private String DOC_SEPARATOR = "\t\t\t";
	private LibSVM model;
	private QueryDocScorer scorer;
	private BM25Scorer bm25scorer;
	private SmallestWindowScorer smScorer;
	private Map<String, Map<String, Integer>> indexMap;

	public PairwiseLearner(boolean isLinearKernel){
		try{
			model = new LibSVM();
			scorer = new QueryDocScorer();
		} catch (Exception e){
			e.printStackTrace();
		}

		if(isLinearKernel){
			model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
		} else {
			model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_RBF, LibSVM.TAGS_KERNELTYPE));
		}
	}

	public PairwiseLearner(double C, double gamma, boolean isLinearKernel){
		try{
			model = new LibSVM();
			scorer = new QueryDocScorer();
		} catch (Exception e){
			e.printStackTrace();
		}

		model.setCost(C);
		model.setGamma(gamma); // only matter for RBF kernel
		if(!isLinearKernel){
			model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_RBF, LibSVM.TAGS_KERNELTYPE));
		} else {
			model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
		}
	}

	@Override
	public Instances extract_train_features(String train_data_file, String train_rel_file,
			Map<String, Double> idfs, Map<String, Double> additionalFeatures) {
		return extract_features(train_data_file, train_rel_file, idfs, additionalFeatures, false);
	}

	private Instances extract_features(String data_file, String rel_file,
			Map<String, Double> idfs, Map<String, Double> additionalFeatures, boolean isTest) {
		Map<Query,List<Document>> trainData = null;
		try {
			trainData = Util.loadTrainData(data_file);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		Map<String, Map<String, Double>> relData = null;

		if (!isTest) {
			try {
				relData = Util.loadRelData(rel_file);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		indexMap = new HashMap<String, Map<String, Integer>>();

		ArrayList<String> labels = new ArrayList<String>();
		labels.add("-1");
		labels.add("1");
		Attribute cls = new Attribute("class", labels);

		// create the dataset
		Instances dataset1 = null;
		Instances dataset2 = null;
		Instances newDataset = null;

		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));

		// add, if present, BM25, smallest window, and pagerank
		if (additionalFeatures.containsKey("bm25")) {
			attributes.add(new Attribute("bm25"));
			bm25scorer = new BM25Scorer(idfs, trainData);
		}

		if (additionalFeatures.containsKey("smallestwindow")) {
			attributes.add(new Attribute("smallestwindow"));
			smScorer = new SmallestWindowScorer(idfs, trainData);
		}

		if (additionalFeatures.containsKey("pagerank")) {
			attributes.add(new Attribute("pagerank"));
		}

		attributes.add(cls);
		dataset1 = new Instances("train_dataset1", attributes, 0);
		dataset2 = new Instances("train_dataset2", attributes, 0);
		newDataset = new Instances("train_dataset3", attributes, 0);

		int currIndex = 0;

		// go through each query
		for (Query q : trainData.keySet()) {
			String queryStr = q.query;

			List<Document> docs = trainData.get(q);

			// doc -> index in dataset
			Map<String, Integer> doc2Index = new HashMap<String, Integer>();

			// go through all pairs of documents for that query
			for (int i = 0; i < docs.size(); i++) {
				for (int j = i + 1; j < docs.size(); j++) {
					Document d1 = trainData.get(q).get(i);
					Document d2 = trainData.get(q).get(j);

					// get the relevance for both docs (if in test mode then these values dont matter)
					double relevance1 = -100;
					double relevance2 = -10;

					if (!isTest) {
						relevance1 = relData.get(queryStr).get(d1.url);
						relevance2 = relData.get(queryStr).get(d2.url);
					}

					// skip this pair if they have the same relevance scores
					if (relevance1 == relevance2) {
						continue;
					}

					// the (query, doc) pairs
					Pair currQueryDoc1 = new Pair(queryStr, d1.url);
					Pair currQueryDoc2 = new Pair(queryStr, d2.url);

					// extract the features from each of the fields in this doc
					Map<String, Double> scores1 = null;
					Map<String, Double> scores2 = null;

					try {
						scores1 = scorer.getSimScore(d1, q, idfs);
						scores2 = scorer.getSimScore(d2, q, idfs);
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}

					// extract the additional features in this doc
					Map<String, Double> additionalFeatures1 = extractAdditionalFeatures(additionalFeatures, d1, q, idfs);
					Map<String, Double> additionalFeatures2 = extractAdditionalFeatures(additionalFeatures, d2, q, idfs);

					// the feature vector for this (query, doc)
					double[] currQueryDocFeatures1 = scorer.constructFeatureArray(scores1, additionalFeatures1, relevance1);
					double[] currQueryDocFeatures2 = scorer.constructFeatureArray(scores2, additionalFeatures2, relevance2);

					if (!isTest) {
						// find the output label for this pair of vectors (-1 or +1)
						String outputLabel = "";
						if (relevance1 > relevance2) {
							outputLabel = "1";
						} else if (relevance1 < relevance2) {
							outputLabel = "-1";
						}

						int index = 5;
						
						if (additionalFeatures.containsKey("bm25")) {
							index++;
						}

						if (additionalFeatures.containsKey("smallestwindow")) {
							index++;
						}

						if (additionalFeatures.containsKey("pagerank")) {
							index++;
						}

						// set the relevance of the resulting feature vector
						currQueryDocFeatures1[index] = dataset1.attribute(index).indexOfValue(outputLabel);
						currQueryDocFeatures2[index] = dataset1.attribute(index).indexOfValue(outputLabel.equals("1") ? "-1" : "1");
					}

					// at this point we have 2 feature vectors: (q, d1) and (q, d2)

					// now need to store the xi and xj separately
					Instance inst1 = new DenseInstance(1.0, currQueryDocFeatures1);
					Instance inst2 = new DenseInstance(1.0, currQueryDocFeatures2);
					dataset1.add(inst1);
					dataset2.add(inst2);

					assert dataset1.size() == dataset2.size();

					if (isTest) {
						doc2Index.put(d1.url + DOC_SEPARATOR + d2.url, currIndex);
						currIndex++;
						doc2Index.put(d2.url + DOC_SEPARATOR + d1.url, currIndex);
						currIndex++;
					}

				} // end j loop

			} // end i loop

			if (isTest) {
				indexMap.put(queryStr, doc2Index);
			}

		} // end query loop

		dataset1.setClassIndex(dataset1.numAttributes() - 1);
		dataset2.setClassIndex(dataset2.numAttributes() - 1);

		// standardize the points between 0 and 1
		Instances newDataset1 = standardizeInstances(dataset1);
		Instances newDataset2 = standardizeInstances(dataset2);

		newDataset1.setClassIndex(newDataset1.numAttributes() - 1);
		newDataset2.setClassIndex(newDataset2.numAttributes() - 1);

		assert newDataset1.size() == newDataset2.size();

		// subtract the 2 datasets (make sure to do xi - xj and xj - xi)
		for (int i = 0; i < newDataset1.size(); i++) {
			Instance i1 = newDataset1.get(i);
			Instance i2 = newDataset2.get(i);

			double[] vector1 = i1.toDoubleArray();
			double[] vector2 = i2.toDoubleArray();

			double relevance1 = vector1[vector1.length - 1];
			double relevance2 = vector2[vector2.length - 1];

			double[] result1 = scorer.subtractVectors(vector1, vector2);
			double[] result2 = scorer.subtractVectors(vector2, vector1);

			// find the output label for this pair of vectors (-1 or +1)
			String outputLabel = "";
			if (relevance1 > relevance2) {
				outputLabel = "1";
			} else if (relevance1 < relevance2) {
				outputLabel = "-1";
			}

			int index = 5;

			if (additionalFeatures.containsKey("bm25")) {
				index++;
			}

			if (additionalFeatures.containsKey("smallestwindow")) {
				index++;
			}

			if (additionalFeatures.containsKey("pagerank")) {
				index++;
			}

			// set the relevance of the resulting feature vector
			result1[index] = newDataset1.attribute(index).indexOfValue(outputLabel);
			result2[index] = newDataset2.attribute(index).indexOfValue(outputLabel.equals("1") ? "-1" : "1");

			newDataset.add(new DenseInstance(1.0, result1));
			newDataset.add(new DenseInstance(1.0, result2));
		}

		newDataset.setClassIndex(newDataset.numAttributes() - 1);

		return newDataset;
	}

	@Override
	public Classifier training(Instances dataset) {
		try {
			model.buildClassifier(dataset);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return model;
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs, Map<String, Double> additionalFeatures) {

		TestFeatures result = new TestFeatures();
		result.features = extract_features(test_data_file, null, idfs, additionalFeatures, true);
		result.index_map = indexMap;

		return result;
	}

	@Override
	public Map<String, List<String>> testing(TestFeatures tf,
			Classifier model) {

		Map<String, List<String>> result = new HashMap<String, List<String>>();

		for (String queryStr : tf.index_map.keySet()) {
			List<Pair<String, Double>> docScores = new ArrayList<Pair<String, Double>>();

			HashSet<String> docsAlreadySeen = new HashSet<String>();

			// for each pair of documents, stores 0 or 1
			Map<Pair<String, String>, Double> pairWiseScores = new HashMap<Pair<String, String>, Double>();

			// url1AndUrl2 is a concatenation, with a delimiter, of url for doc1 and url for doc2
			for (String url1AndUrl2 : tf.index_map.get(queryStr).keySet()) {
				String[] urls = url1AndUrl2.split(DOC_SEPARATOR);
				String doc1Url = urls[0];
				String doc2Url = urls[1];

				// get the features for this testing point
				Instance i = tf.features.get(tf.index_map.get(queryStr).get(url1AndUrl2));

				// will be 0.0 or 1.0
				double predictedClass = -1;
				try {
					predictedClass = model.classifyInstance(i);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}

				pairWiseScores.put(new Pair(doc1Url, doc2Url), predictedClass);
				pairWiseScores.put(new Pair(doc2Url, doc1Url), predictedClass == 0.0 ? 1.0 : 0.0);

				docsAlreadySeen.add(doc1Url);
				docsAlreadySeen.add(doc2Url);
			}

			List<String> documents = new ArrayList<String>();
			for (String doc : docsAlreadySeen) {
				documents.add(doc);
			}

			// get the docs, sorted by relevance
			List<String> docs = this.getSortedDocsPairwise(documents, pairWiseScores);
			result.put(queryStr, docs);
		}

		return result;
	}

	/*
	 * Returns a the feature vector, represented as a map of string -> score.
	 */
	private Map<String, Double> extractAdditionalFeatures(Map<String, Double> additionalFeatures, Document d,
			Query q, Map<String, Double> idfs) {

		if (additionalFeatures.containsKey("bm25")) {
			try {
				additionalFeatures.put("bm25", bm25scorer.getBM25Score(d, q, idfs));
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		if (additionalFeatures.containsKey("smallestwindow")) {
			try {
				additionalFeatures.put("smallestwindow", smScorer.getSmallestWindow(d, q, idfs));
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		if (additionalFeatures.containsKey("pagerank")) {
			additionalFeatures.put("pagerank", (double) d.page_rank);
		}

		return additionalFeatures;

	}

	private Instances standardizeInstances(Instances dataset) {
		Standardize filter = new Standardize();
		Instances newDataset = null;

		try {
			filter.setInputFormat(dataset);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		try {
			newDataset = Filter.useFilter(dataset, filter);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		return newDataset;
	}

}
