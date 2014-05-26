package cs276.pa4;

import java.util.HashMap;
import java.util.Map;

public class QueryDocScorer
{
	private String[] TFTYPES = {"url","title","body","header","anchor"};
	private double smoothingBodyLength = 30000.0;
	private int NUM_DOCS = 98998;
	

	/*
	 * Returns a map of field type -> score for that field
	 */
	public Map<String, Double> getScores(Map<String, Map<String, Double>> tfs, Query q, 
			Map<String, Double> tfQuery, Document d, Map<String, Double> idfs) throws Exception
			{
		Map<String, Double> scores = new HashMap<String, Double>();

		for (String type : tfs.keySet()) {
			double currTotal = 0;
			for (String term : tfs.get(type).keySet()) {
				if (!tfQuery.containsKey(term)) {
					throw new Exception("Exception in getScores(): KEYS ARE NOT THE SAME!");
				}		

				double idfComponent = -1;
				if (idfs.containsKey(term)) {
					idfComponent = Math.log10(NUM_DOCS + idfs.size()) - Math.log10(idfs.get(term) + 1.0);
				} else {
					idfComponent = Math.log10(idfs.size() + NUM_DOCS);
				}

				currTotal += tfs.get(type).get(term) * tfQuery.get(term) * idfComponent;
			}

			scores.put(type, currTotal);
		}

		return scores;
			}


	public void normalizeTFs(Map<String, Map<String, Double>> tfs, Document d, Query q)
	{
		for (String type : tfs.keySet()) {			
			for (String term : tfs.get(type).keySet()) {
				double tf = tfs.get(type).get(term);
				double newVal = 0;
				if (tf != 0) {
					newVal = (1.0 + Math.log10(tf)) / (this.smoothingBodyLength + d.body_length);
				}
				tfs.get(type).put(term, newVal);
			}
		}
	}


	public Map<String, Double> getSimScore(Document d, Query q, Map<String,Double> idfs) throws Exception 
	{
		Map<String,Map<String, Double>> tfs = this.getDocTermFreqs(d, q);
		this.normalizeTFs(tfs, d, q);

		Map<String,Double> tfQuery = getQueryFreqs(q);

		return getScores(tfs, q, tfQuery, d, idfs);
	}

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
		Map<String, Map<String, Double>> tfs = new HashMap<String,Map<String, Double>>();

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

}
