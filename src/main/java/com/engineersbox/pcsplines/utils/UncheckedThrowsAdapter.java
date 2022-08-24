package com.engineersbox.pcsplines.utils;

import java.util.function.Function;

public interface UncheckedThrowsAdapter {

    public interface ThrowsFunction<T, R> {
        R apply(final T t) throws Exception;
    }

    public static <T, R> Function<T, R> unchecked(final ThrowsFunction<T, R> function) {
        return (final T t) -> {
            try {
                return function.apply(t);
            } catch (final Exception e) {
                throw new RuntimeException(e);
            }
        };
    }

}
