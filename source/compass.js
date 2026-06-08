
import * as base from './base.js';

const compass = {};
compass.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension == 'def' || extension == 'txt') {
            return context.set('compass.def');
        }
        if (extension == 'bin' || extension == 'cbin') {
            return context.set('compass.bin');
        }
        return undefined;
    }

    async open(context) {
        const metadata = await compass.Metadata.open(context);
        const identifier = context.identifier.toLowerCase();
        const openText = (param, bin) => {
            const reader = new compass.TextParamReader(metadata, param, bin);
            return new compass.Model(metadata, reader.net);
        };
        let bin = null;
        let content = null;

        switch (context.type) {
            case 'compass.def': {
                if (identifier.endsWith('.def') || identifier.endsWith('.txt')) {
                    bin = context.identifier.substring(0, context.identifier.length - 3) + 'bin';
                }
                let content = null;
                const text = await context.read('text');
                try {
                    content = await context.fetch(bin);
                    return openText(text, content);
                } catch {
                    return openText(text, null);
                }
            }
            case 'compass.bin': {
                if (identifier.endsWith('.bin')) {
                    bin = context.identifier.substring(0, context.identifier.length - 3);
                }
                if (identifier.endsWith('.cbin')) {
                    bin = context.identifier.substring(0, context.identifier.length - 4);
                }
                try {
                    const content = await context.fetch(bin + "def");
                    const text = await content.read('text');
                    return openText(text, context.stream.peek());

                } catch (error) {
                    try {
                        const stream = await context.fetch(bin + "txt");
                        const text = await stream.read('text');
                        return openText(text, context.stream.peek());

                    } catch (error) {
                        return openText(null, context.stream.peek());

                    }
                }

            }
        }
    }

};

compass.Model = class {

    constructor(metadata, param) {
        this._subgraphs = param.subgraphs.map(graph => new compass.Graph(metadata, graph));
        this._graphs = [
            new compass.Graph(metadata, param), ...this._subgraphs
        ];
        this.process_subgraph(metadata);
        this.modules = this._graphs;
    }

    process_subgraph(metadata) {
        let sgmap = {};
        for (let i = 0; i < this._subgraphs.length; i++) {
            sgmap[this._subgraphs[i].name] = this._subgraphs[i];
        }
        for (let i = 0; i < this._graphs.length; i++) {
            for (let node of this._graphs[i]._nodes) {
                for (let j = 0; j < node._attributes.length; j++) {
                    if (node._attributes[j].type === 'graph') {
                        let attr = node._attributes[j];
                        node._attributes[j]._value = sgmap[attr.value];
                    }
                }
            }
        };

    }

    get format() {
        return 'Compass';
    }

    get graphs() {
        return this._graphs;
    }
};

compass.Graph = class {

    constructor(metadata, net) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        this.name = net["name"];
        if ("subgraph_name" in net)
            this.name = net["subgraph_name"];
        for (const layer of net.layers) {
            if (layer.type == 'Input') {
                const input = new compass.Parameter(layer.name,
                    layer.outputs.map((output) => new compass.Argument(output.name,
                        new compass.TensorType(output.type, output.shape), null)));
                this._inputs.push(input);
            }
            else {
                const node = new compass.Node(metadata, layer);
                this._nodes.push(node);
                for (const t of layer.outputs) {
                    if (net.output_tensors.includes(t.name)) {
                        const output = new compass.Parameter(t.name,
                            [t].map((output) => new compass.Argument(output.name,
                                new compass.TensorType(output.type, output.shape), null)));
                        this._outputs.push(output);
                    }
                }
            }
        }
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get nodes() {
        return this._nodes;
    }
};


compass.Parameter = class {

    constructor(name, args) {
        this._name = name;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return true;
    }

    get arguments() {
        return this._arguments;
    }

    get value() {
        return this._arguments;

    }
};


compass.Argument = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new compass.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get initializer() {
        return this._initializer;
    }
    get value() {
        return this._initializer;
    }
};

compass.Node = class {

    constructor(metadata, layer) {
        this._chain = [];
        this._attributes = [];
        this._name = layer.name;
        let type = layer.type;
        this._type = metadata.type(type) || { name: type };

        let initializers = [];

        for (const name of Object.keys(layer.param)) {
            const value = layer.param[name];
            const attribute = new compass.Attribute(metadata.attribute(type, name), name, value);
            this._attributes.push(attribute);

        }
        // initializers = layer.layer.blobs.map((blob) => new compass.Tensor(blob));

        this._inputs = layer.inputs.map((t) => new compass.Parameter(t.name, [
            new compass.Argument(t.name, new compass.TensorType(t.type, t.shape, t.scale, t.zp), null),


        ]));
        this._weights = layer.weights.map((t) => new compass.Parameter(t.name, [new compass.Argument(t.name,
            new compass.Tensor(new compass.TensorType(t.type, t.shape), t.data, "Weight")
            , null)]));
        this._outputs = layer.outputs.map((t) => new compass.Parameter(t.name, [new compass.Argument(t.name, new compass.TensorType(t.type, t.shape, t.scale, t.zp), null)]));

    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get inputs() {
        return this._inputs.concat(this._weights);
    }

    get outputs() {
        return this._outputs;
    }

    get weights() {
        return this._weights;
    }
    get attributes() {
        return this._attributes;
    }

    get chain() {
        return this._chain;
    }
};

compass.Attribute = class {

    constructor(metadata, name, value, defaultValue) {
        this._name = name;
        this._value = value;
        if (metadata && metadata.type) {
            this._type = metadata.type;
        }
        if (value instanceof compass.TensorShape) {
            this._type = 'shape';
        }
        if (metadata && Object.prototype.hasOwnProperty.call(metadata, 'visible') && !metadata.visible) {
            this._visible = false;
        }
        if (metadata && Object.prototype.hasOwnProperty.call(metadata, 'default')) {
            defaultValue = metadata.default;
        }
        if (defaultValue !== undefined) {
            if (this._value == defaultValue) {
                this._visible = false;
            }
            else if (Array.isArray(this._value) && Array.isArray(defaultValue)) {
                if (this._value.length == defaultValue.length &&
                    this._value.every((item, index) => { return item == defaultValue[index]; })) {
                    this._visible = false;
                }
            }
        }
        if (this._type) {
            this._value = compass.Utility.enum(this._type, this._value);
        }
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};


compass.Tensor = class {

    constructor(type, data, kind) {
        this._type = type;
        this._data = data;
        this._kind = kind;
    }

    get kind() {
        return this._kind;
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._context().state || null;
    }

    get value() {
        const context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        const context = this._context();
        if (context.state) {
            return this._type.toString();
        }
        context.limit = 10000;
        const value = this._decode(context, 0);
        return this._type.toString() + "\n" + JSON.stringify(value, null, 4);
    }

    _context() {
        const context = {};
        context.index = 0;
        context.count = 0;
        context.state = null;

        if (this._type.dataType == '?') {
            context.state = 'Tensor has unknown data type.';
            return context;
        }
        if (!this._type.shape || (this._type.shape.dimensions && this._type.shape.dimensions.length == 0)) {
            context.state = 'Tensor has no dimensions.';
            return context;
        }

        if (!this._data) {
            context.state = 'Tensor data is empty.';
            return context;
        }

        switch (this._type.dataType) {
            case 'int8':
            case 'uint8':
            case 'int16':
            case 'uint16':
            case 'int32':
            case 'uint32':
            case 'int64':
            case 'uint64':
            case 'float16':
            case 'bfloat16':
            case 'float32':
            case 'float64':
            case 'aligned_int4':
            case 'aligned_uint4':
            case 'aligned_int12':
            case 'aligned_uint12':
            case 'float8_e4m3fn':
            case 'float8_e5m2':
            case 'float4_e2m1fn':
                context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
                break;
            default:
                context.state = 'Tensor data type is not implemented.';
                break;
        }

        context.dataType = this._type.dataType;
        context.shape = this._type.shape.dimensions;
        return context;
    }

    _decode(context, dimension) {
        const shape = context.shape.length == 0 ? [1] : context.shape;
        const results = [];
        const size = shape[dimension];
        const temp_buffer = new ArrayBuffer(4);
        const temp_view = new DataView(temp_buffer);
        const fp4e2m1_table = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6];
        const fp8e4m3_table = [0.0, 0.001953125, 0.00390625, 0.005859375, 0.0078125, 0.009765625, 0.01171875,
                                0.013671875, 0.015625, 0.017578125, 0.01953125, 0.021484375, 0.0234375, 0.025390625, 0.02734375,
                                0.029296875, 0.03125, 0.03515625, 0.0390625, 0.04296875, 0.046875, 0.05078125, 0.0546875, 0.05859375,
                                0.0625, 0.0703125, 0.078125, 0.0859375, 0.09375, 0.1015625, 0.109375, 0.1171875, 0.125, 0.140625,
                                0.15625, 0.171875, 0.1875, 0.203125, 0.21875, 0.234375, 0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625,
                                0.4375, 0.46875, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.125, 1.25, 1.375, 1.5,
                                1.625, 1.75, 1.875, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
                                8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0,
                                36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 64.0, 72.0, 80.0, 88.0, 96.0, 104.0, 112.0, 120.0, 128.0,
                                144.0, 160.0, 176.0, 192.0, 208.0, 224.0, 240.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0];
        const fp8e5m2_table = [0.0, 1.52587890625e-05, 3.0517578125e-05, 4.57763671875e-05, 6.103515625e-05, 7.62939453125e-05, 9.1552734375e-05,
                                0.0001068115234375, 0.0001220703125, 0.000152587890625, 0.00018310546875, 0.000213623046875,
                                0.000244140625, 0.00030517578125, 0.0003662109375, 0.00042724609375, 0.00048828125, 0.0006103515625,
                                0.000732421875, 0.0008544921875, 0.0009765625, 0.001220703125, 0.00146484375, 0.001708984375, 0.001953125,
                                0.00244140625, 0.0029296875, 0.00341796875, 0.00390625, 0.0048828125, 0.005859375, 0.0068359375, 0.0078125,
                                0.009765625, 0.01171875, 0.013671875, 0.015625, 0.01953125, 0.0234375, 0.02734375, 0.03125, 0.0390625, 0.046875,
                                0.0546875, 0.0625, 0.078125, 0.09375, 0.109375, 0.125, 0.15625, 0.1875, 0.21875, 0.25, 0.3125, 0.375, 0.4375,
                                0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0,
                                16.0, 20.0, 24.0, 28.0, 32.0, 40.0, 48.0, 56.0, 64.0, 80.0, 96.0, 112.0, 128.0, 160.0, 192.0, 224.0, 256.0,
                                320.0, 384.0, 448.0, 512.0, 640.0, 768.0, 896.0, 1024.0, 1280.0, 1536.0, 1792.0, 2048.0, 2560.0, 3072.0,
                                3584.0, 4096.0, 5120.0, 6144.0, 7168.0, 8192.0, 10240.0, 12288.0, 14336.0, 16384.0, 20480.0, 24576.0,
                                28672.0, 32768.0, 40960.0, 49152.0, 57344.0,
                                Infinity, NaN, NaN, NaN];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (this._type.dataType) {
                    case 'float64':
                        results.push(context.data.getFloat64(context.index, true));
                        context.index += 8;
                        context.count++;
                        break;
                    case 'float32':
                        results.push(context.data.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'float16':
                        results.push(context.data.getFloat16(context.index, true));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'int8':
                    case 'aligned_int4':
                        results.push(context.data.getInt8(context.index, true));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'uint8':
                    case 'aligned_uint4':
                        results.push(context.data.getUint8(context.index, true));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'int32':
                        results.push(context.data.getInt32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'uint32':
                        results.push(context.data.getUint32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'int16':
                    case 'aligned_int12':
                        results.push(context.data.getInt16(context.index, true));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'uint16':
                    case 'aligned_uint12':
                        results.push(context.data.getUint16(context.index, true));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'bfloat16':
                        temp_view.setUint32(0, context.data.getUint16(context.index, true)<<16, true);
                        results.push(temp_view.getFloat32(0,true));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'float8_e4m3fn':
                        var v = context.data.getInt8(context.index, true);
                        var r = fp8e4m3_table[v&0x7f];
                        if(v>>7)
                        {
                            r=-r;
                        }
                        results.push(r);
                        context.index += 1;
                        context.count++;
                    case 'float8_e5m2':
                        var v = context.data.getInt8(context.index, true);
                        var r = fp8e5m2_table[v&0x7f];
                        if(v>>7)
                        {
                            r=-r;
                        }
                        results.push(r);;
                        context.index += 1;
                        context.count++;
                            break;
                    case 'float4_e2m1fn':
                        results.push(fp4e2m1_table[context.data.getUint8(context.index, true)]);
                        context.index += 1;
                        context.count++;
                }
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dimension + 1));
            }
        }
        if (context.shape.length == 0) {
            return results[0];
        }
        return results;
    }
};
compass.TensorType = class {

    constructor(dataType, shape, scale = null, zp = null) {
        this._dataType = dataType;
        this._shape = shape;
        this._scale = scale;
        this._zp = zp;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        var ret = "";
        if (this._scale != null && (this._scale != 1.0 || this._zp != 0)) {
            ret = " scale:" + this._scale.toString() + ", zp:" + this._zp.toString();
        }
        return (this.dataType || '?') + this._shape.toString() + ret;
    }
};

compass.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return this._dimensions ? ('[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']') : '';
    }
};

compass.Utility = class {

    static layerType(type) {
        type = type || 0;
        if (!compass.Utility._layerTypeMap) {
            compass.Utility._layerTypeMap = new Map();
            const known = { 'BNLL': 'BNLL', 'HDF5': 'HDF5', 'LRN': 'LRN', 'RELU': 'ReLU', 'TANH': 'TanH', 'ARGMAX': 'ArgMax', 'MVN': 'MVN', 'ABSVAL': 'AbsVal' };
            for (const key of Object.keys(compass.proto.V1LayerParameter.LayerType)) {
                const value = compass.proto.V1LayerParameter.LayerType[key];
                compass.Utility._layerTypeMap.set(value, key.split('_').map((item) => known[item] || item.substring(0, 1) + item.substring(1).toLowerCase()).join(''));
            }
        }
        return compass.Utility._layerTypeMap.has(type) ? compass.Utility._layerTypeMap.get(type) : type.toString();
    }

    static enum(name, value) {
        let type = compass.proto;
        const parts = name.split('.');
        while (type && parts.length > 0) {
            type = type[parts.shift()];
        }
        if (type) {
            compass.Utility._enumKeyMap = compass.Utility._enumKeyMap || new Map();
            if (!compass.Utility._enumKeyMap.has(name)) {
                const map = new Map(Object.entries(type).map((pair) => [pair[1], pair[0]]));
                compass.Utility._enumKeyMap.set(name, map);
            }
            const map = compass.Utility._enumKeyMap.get(name);
            if (map.has(value)) {
                return map.get(value);
            }
        }
        return value;
    }
};

compass.Metadata = class {

    static async open(context) {
        if (!compass.Metadata._metadata) {
            let data = null;
            try {
                data = await context.asset('compass-metadata.json');
            } catch {
                // continue regardless of error
            }
            compass.Metadata._metadata = new compass.Metadata(data);
        }
        return compass.Metadata._metadata;
    }

    constructor(data) {
        this._map = new Map();
        this._attributeCache = new Map();
        if (data) {
            const metadata = JSON.parse(data);
            this._map = new Map(metadata.map((item) => [item.name, item]));
        }
    }

    type(name) {
        return this._map.get(name);
    }

    attribute(type, name) {
        const key = type + ':' + name;
        if (!this._attributeCache.has(key)) {
            this._attributeCache.set(key, null);
            const metadata = this.type(type);
            if (metadata && Array.isArray(metadata.attributes) && metadata.attributes.length > 0) {
                for (const attribute of metadata.attributes) {
                    this._attributeCache.set(type + ':' + attribute.name, attribute);
                }
            }
        }
        return this._attributeCache.get(key);
    }
};

compass.TextParamReader = class {

    constructor(metadata, buffer, weights) {
        const reader = buffer;
        const sections = []
        let lines = {};
        for (; ;) {
            const line = reader.read('\n');
            if (line === undefined) {
                if (Object.keys(lines).length != 0) {
                    sections.push(lines);

                }
                break;
            }
            if (line.trim().length == 0 && Object.keys(lines).length != 0) {
                sections.push(lines);
                lines = {};
            }
            if (line.trim().length > 0) {
                const kv = line.split('=').filter((s) => s.length != 0);
                const key = kv.shift();
                const value = kv.shift();
                lines[key] = value;
            }
        }
        const net = sections.shift();
        const header_meta = metadata.type("header");
        if (header_meta) {
            for (const k of header_meta["require"]) {
                if (!Object.keys(net).includes(k)) {
                    throw new compass.Error("Missing required field '" + JSON.stringify(k) + "'.");
                }
            }
        }
        for (const k of Object.keys(net)) {
            net[k] = this.parse_param(net[k]);
        }
        net.layers = [];
        net.subgraphs = [];
        this.parse_sections(net, sections, weights, net["layer_number"]);
        this._net = net;
    }

    parse_sections(root, sections, weights, layer_number) {
        let stack = [];
        let layer_numbers = [];
        let net = root;
        for (let i = 0; i < sections.length; i++) {
            let section = sections[i];
            if ("subgraph_name" in section) {
                if (!("layer_number" in section)) {
                    throw new compass.Error("Missing required field '" + JSON.stringify(layer_number) + "' of subgraph header.");
                }
                layer_numbers.push(layer_number);
                stack.push(net);
                layer_number = parseInt(section["layer_number"]);
                const sg = sections[i];
                sg.layers = [];
                sg.subgraphs = [];
                root.subgraphs.push(sg);
                net = sg;
                continue;
            }
            net.layers.push(this.parse_layer(section, weights));
            if (net.layers.length == layer_number) {
                net = stack.pop();
                layer_number = layer_numbers.pop();
            }
        }
        return net;

    }

    parse_param(s) {
        if (s == "[]" || s == "") {
            return "";
        }
        if (s.toLowerCase() == "true") {
            return true;
        }
        if (s.toLowerCase() == "false") {
            return false;
        }
        if ([].includes(s)) {
            return s;
        }
        if (!isNaN(s) && !isNaN(parseFloat(s))) {
            return parseFloat(s);
        }
        if (s.includes(",") || s.includes("[")) {
            const stack = [];
            const ret = []
            let current = ret;
            let idx = 0;
            let begin = 0;
            while (idx < s.length) {
                if (s[idx] == '[') {
                    begin = idx + 1;
                    const n = [];
                    current.push(n);
                    stack.push(current);
                    current = n;

                }
                else if (s[idx] == "]") {
                    const subs = s.substring(begin, idx).trim();
                    if (subs.length) {
                        current.push(this.parse_param(subs));
                    }
                    begin = idx + 1;
                    current = stack.pop();
                }
                else if (s[idx] == ",") {
                    const subs = s.substring(begin, idx).trim();
                    if (subs.length) {
                        current.push(this.parse_param(subs));
                    }
                    begin = idx + 1;
                }
                idx++;
            }
            if (ret.length == 1) {
                return ret[0];
            }
            return ret;
        }
        return s;
    }
    parse_tensor(section, prefix) {
        const ret = [];
        const names = this.parse_param(section[prefix]);
        const types = this.parse_param(section[prefix + "_type"]);
        const shapes = this.parse_param(section[prefix + "_shape"]);
        let scales = [];
        let zps = [];
        if (prefix + "_scale" in section) {
            scales = this.parse_param(section[prefix + "_scale"]);
        }
        if (prefix + "_zp" in section) {

            zps = this.parse_param(section[prefix + "_zp"]);
        }
        for (let i = 0; i < names.length; ++i) {
            const t = {};
            t.name = String(names[i]);
            t.type = String(types[i]);
            t.shape = new compass.TensorShape(shapes[i]);
            t.scale = 1.;
            t.zp = 0.;
            if (scales.length > i) {
                t.scale = scales[i];
            }
            if (zps.length > i) {
                t.zp = zps[i];
            }
            ret.push(t);
        }
        return ret;

    }
    parse_layer(section, weights) {
        const layer = {};
        layer.type = String(section["layer_type"]);
        layer.name = String(section["layer_name"]);
        layer.id = section["layer_id"];
        layer.outputs = this.parse_tensor(section, "layer_top");
        layer.inputs = this.parse_tensor(section, "layer_bottom");
        layer.weights = [];
        layer.param = {};
        for (const k of Object.keys(section)) {
            if (!k.startsWith("layer_")) {
                layer.param[k] = this.parse_param(section[k]);
            }
        }
        for (const k of Object.keys(layer.param)) {
            if (k.endsWith("_shape")) {
                const shape = layer.param[k];
                const type = k.replace("_shape", "_type");
                const offset = k.replace("_shape", "_offset");
                const s = k.replace("_shape", "_size");
                const t = {};
                // t.name = k.replace("_shape", "");
                t.type = layer.param[type];
                t.shape = new compass.TensorShape(shape);
                t.offset = layer.param[offset];
                t.size = layer.param[s];
                t.name = k.replace("_shape", "")
                t.scale = 1.;
                t.zp = 0.;
                t.data = null;
                if (weights) {
                    t.data = weights.subarray(t.offset, t.offset + t.size);
                }
                delete layer.param[k];
                delete layer.param[type];
                delete layer.param[offset];
                delete layer.param[s];
                layer.weights.push(t);
            }
            else if (k.endsWith("_value") && Object.keys(layer.param).includes(k.replace("_value", "_type"))) {
                const t = {};
                t.type = layer.param[k.replace("_value", "_type")];
                t.value = layer.param[k];
                layer.param[k.replace("_value", "")] = t;
                delete layer.param[k];
                delete layer.param[k.replace("_value", "_type")];
            }
            else {
                for (const prefix of ["kernel", "pad", "stride", "dilation"]) {
                    if (k.startsWith(prefix + "_")) {
                        let v = layer.param[prefix];
                        if (!v) {
                            v = {};
                        }
                        v[k.replace(prefix + "_", "")] = layer.param[k];
                        delete layer.param[k];
                        layer.param[prefix] = v;
                    }
                }
            }
        }
        return layer;
    }
    get net() {
        return this._net;
    }
};

compass.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Compass model.';
    }
};

export const ModelFactory = compass.ModelFactory;