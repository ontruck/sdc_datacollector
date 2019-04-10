/*
 * Copyright 2018 StreamSets Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.streamsets.pipeline.stage.processor.tensorflow.typesupport;

import com.streamsets.pipeline.api.Field;
import com.streamsets.pipeline.api.impl.Utils;
import org.tensorflow.DataType;
import org.tensorflow.Tensor;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

final class StringTensorTypeSupport extends AbstractTensorDataTypeSupport<ByteBuffer, String> {

    @Override
    public ByteBuffer allocateBuffer(long[] shape) {
        return ByteBuffer.allocate(calculateCapacityForShape(shape));
    }

    @Override
    public Tensor<String> createTensor(long[] shape, ByteBuffer buffer) {
        return Tensor.create(String.class, shape, buffer);
    }

    @Override
    public void writeField(ByteBuffer buffer, Field field) {
        Utils.checkState(field.getType() == Field.Type.DOUBLE, "Not a double scalar");
        buffer.put(field.getValueAsByte());
    }

    @Override
    public List<Field> createListField(Tensor<String> tensor, ByteBuffer byteBuffer) {
        List<Field> fields = new ArrayList<>();
        tensor.writeTo(byteBuffer);
        byte[] bytes = byteBuffer.array();
        for (byte aByte : bytes) {
            fields.add(Field.create(aByte));
        }
        return fields;
    }

    @Override
    public Field createPrimitiveField(Tensor<String> tensor) {
        return Field.create(tensor.doubleValue());
    }

    @Override
    public DataType getDataType() {
        return DataType.STRING;
    }
}
