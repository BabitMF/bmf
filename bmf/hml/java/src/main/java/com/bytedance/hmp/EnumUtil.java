package com.bytedance.hmp;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

public class EnumUtil {
    public static Object fromValue(Class cls, Object v){
        Object[] enumConstants = cls.getEnumConstants();
        for (Object object: enumConstants){
            Class<?> aClass = object.getClass();
            Method getValue = null;
            try {
                getValue = aClass.getMethod("getValue");
            } catch (NoSuchMethodException e) {
                e.printStackTrace();
            }
            try {
                if(getValue.invoke(object).equals(v)){
                    return object;
                }
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            } catch (InvocationTargetException e) {
                e.printStackTrace();
            }
        }
        return null;
    }
}