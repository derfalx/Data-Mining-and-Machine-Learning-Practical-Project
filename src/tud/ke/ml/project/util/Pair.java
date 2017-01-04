package tud.ke.ml.project.util;

public class Pair<A, B> {
	protected A a;
	protected B b;

	public Pair(A a, B b) {
		this.a = a;
		this.b = b;
	}

	public A getA() {
		return a;
	}

	public void setA(A a) {
		this.a = a;
	}

	public B getB() {
		return b;
	}

	public void setB(B b) {
		this.b = b;
	}

	@Override
	public String toString() {
		return "(" + a + "," + b + ")";
	}

	@Override
	public boolean equals(Object p2) {
		return toString().equals(p2.toString());
	}

	@Override
	public int hashCode() {
		return toString().hashCode();
	}
}
