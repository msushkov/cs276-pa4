package cs276.pa4;

import java.util.HashMap;
import java.util.Map;

import weka.core.Attribute;

public abstract class Scorer 
{

	public int NUM_DOCS = 98998;
	
	Map<String,Double> idfs;
	
	String[] TFTYPES = {"url","title","body","header","anchor"};

	//handle the query vector
	public Map<String,Double> getQueryFreqs(Query q)
	{
		Map<String, Double> tfQuery = new HashMap<String, Double>();

		for (String term : q.words) {
			term = term.toLowerCase().replaceAll("[^A-Za-z0-9 ]", "");

			if (tfQuery.containsKey(term)) {
				tfQuery.put(term, tfQuery.get(term) + 1.0);
			} else {
				tfQuery.put(term, 1.0);
			}
		}

		return tfQuery;
	}


	/*/
	 * Creates the various kinds of term frequencies (url, title, body, header, and anchor)
	 * You can override this if you'd like, but it's likely that your concrete classes will share this implementation
	 */
	public Map<String,Map<String, Double>> getDocTermFreqs(Document d, Query q)
	{
		//map from tf type -> queryWord -> score
		Map<String,Map<String, Double>> tfs = new HashMap<String,Map<String, Double>>();

		//////////handle counts//////

		// go through url, title, etc
		for (String type : TFTYPES) {
			Map<String, Double> currTermMap = new HashMap<String, Double>();
			tfs.put(type, currTermMap);

			//loop through query terms increasing relevant tfs
			for (String queryWord : q.words) {
				queryWord = queryWord.toLowerCase().replaceAll("[^A-Za-z0-9 ]", "");

				// everything by default has a value of 0
				currTermMap.put(queryWord, 0.0);

				// url
				if (type.equals("url") && d.url != null) {
					double numInUrl = countNumOfOccurrencesInUrl(queryWord, d.url);
					currTermMap.put(queryWord, numInUrl);
				}

				// title
				if (type.equals("title") && d.title != null) {
					double numInTitle = countNumOfOccurrencesInString(queryWord, d.title);
					currTermMap.put(queryWord, numInTitle);
				}

				// headers
				if (type.equals("header") && d.headers != null) {
					// create a single string out of the headers list
					StringBuffer concatenatedHeader = new StringBuffer();
					for (String header : d.headers) {
						concatenatedHeader.append(header);
						concatenatedHeader.append(" ");
					}

					double numInHeader = countNumOfOccurrencesInString(queryWord, concatenatedHeader.toString());
					currTermMap.put(queryWord, numInHeader);
				}

				// body
				if (type.equals("body") && d.body_hits != null) {
					if (d.body_hits.containsKey(queryWord)) {
						currTermMap.put(queryWord, (double) d.body_hits.get(queryWord).size());
					}
				}

				// anchor
				if (type.equals("anchor") && d.anchors != null) {
					int count = 0;
					for (String anchor : d.anchors.keySet()) {
						// for each anchor, count how many times the query term occurs in that anchor and multiply by the number of times that anchor occurs
						count += countNumOfOccurrencesInString(queryWord, anchor) * d.anchors.get(anchor);
					}

					currTermMap.put(queryWord, (double) count);
				}

			} // end loop over terms

		} // end loop over types

		return tfs;
	}

	/*
	 * Counts the number of occurrences of term in url.
	 */
	private double countNumOfOccurrencesInUrl(String term, String url) {
		String[] words = url.toLowerCase().split("[^A-Za-z0-9 ]");	    
		double count = 0;
		for (String w : words) {
			if (term.equals(w)) {
				count++;
			}
		}
		return count;
	}

	/*
	 * Counts the number of occurrences of term in str.
	 */
	private double countNumOfOccurrencesInString(String term, String str) {
		str = str.toLowerCase().replaceAll("[^A-Za-z0-9 ]", "");
		String[] words = str.split(" ");

		double count = 0;
		for (String w : words) {
			if (term.equals(w)) {
				count++;
			}
		}
		return count;
	}

	/*
	 * Input: map of field type -> score for that type
	 * Returns: the feature vector with the relevance score as a double[]
	 * features (in order): "url","title","body","header","anchor"
	 */
	public double[] constructFeatureArray(Map<String, Double> fieldTypeMap, Map<String, Double> additionalFeatures, double relevance) {
		double[] result = new double[6 + additionalFeatures.size()];

		result[0] = fieldTypeMap.get("url");
		result[1] = fieldTypeMap.get("title");
		result[2] = fieldTypeMap.get("body");
		result[3] = fieldTypeMap.get("header");
		result[4] = fieldTypeMap.get("anchor");
		
		int index = 5;
		
		if (additionalFeatures.containsKey("bm25")) {
			result[index] = additionalFeatures.get("bm25");
			index++;
		}

		if (additionalFeatures.containsKey("smallestwindow")) {
			result[index] = additionalFeatures.get("smallestwindow");
			index++;
		}

		if (additionalFeatures.containsKey("pagerank")) {
			result[index] = additionalFeatures.get("pagerank");
			index++;
		}
		
		result[index] = relevance;

		return result;
	}

	/*
	 * Subtract 2 arrays and return the result.
	 */
	public double[] subtractVectors(double[] a, double[] b) {
		assert a.length == b.length;

		double[] result = new double[a.length];

		for (int i = 0; i < a.length; i++) {
			result[i] = a[i] - b[i];
		}

		return result;
	}

	private void debugPrinttfResult(Map<String,Map<String, Double>> tf) {
		for (String type : tf.keySet()) {
			System.out.print("TYPE: " + type + ": ");

			for (String term : tf.get(type).keySet()) {
				System.out.print(term + " " +  tf.get(type).get(term) + ", ");
			}

			System.out.println();
		}

		System.out.println("end query...\n\n\n\n");
	}

}
