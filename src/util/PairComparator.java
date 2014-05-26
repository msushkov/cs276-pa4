package util;

import java.util.Comparator;

public class PairComparator implements Comparator<Pair> {

	/*
	 * Compare using pair.getSecond()
	 */
	@Override
	public int compare(Pair a, Pair b) {
		return (Double) a.getSecond() < (Double) b.getSecond() ? -1 : (Double) a.getSecond() == (Double) b.getSecond() ? 0 : 1;
	}
	
}
