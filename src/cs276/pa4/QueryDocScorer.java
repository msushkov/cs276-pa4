package cs276.pa4;

import java.util.HashMap;
import java.util.Map;

public class QueryDocScorer extends Scorer
{
	protected String[] TFTYPES = {"url","title","body","header","anchor"};
	private double smoothingBodyLength = 30000.0;

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
					//newVal = (1.0 + Math.log10(tf)) / (1 + Math.log10(d.body_length));
				}
				tfs.get(type).put(term, newVal);
			}
		}
	}

	public Map<String, Double> getSimScore(Document d, Query q, Map<String, Double> idfs) throws Exception 
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
}
