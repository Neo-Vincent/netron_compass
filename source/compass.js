
var compass = compass || {};
var protobuf = protobuf || require('./protobuf');

compass.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension == 'def' || extension == 'txt') {
            return 'compass.def';
        }
        return undefined;
    }

    open(context, match) {
        return compass.Metadata.open(context).then((metadata) => {
            const identifier = context.identifier.toLowerCase();
            const openText = (param, bin) => {
                const reader = new compass.TextParamReader(metadata, param);
                return new compass.Model(metadata, reader.net, bin);
            };
            let bin = null;
            switch (match) {
                case 'compass.def': {
                    if (identifier.endsWith('.def')) {
                        bin = context.identifier.substring(0, context.identifier.length - 3) + '.bin';
                    }
                    return context.request(identifier, null).then((stream) => {
                        const buffer = stream.read();
                        return openText(context.stream.peek(), buffer);
                    }).catch(() => {
                        return openText(context.stream.peek(), null);
                    });
                }
            }
        });
    }

};

compass.Model = class {

    constructor(metadata, param, bin) {
        this._graphs = [
            new compass.Graph(metadata, param, bin)
        ];
    }

    get format() {
        return 'Compass';
    }

    get graphs() {
        return this._graphs;
    }
};

compass.Graph = class {

    constructor(metadata, net, bin) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        // const blobReader = new ncnn.BlobReader(bin);
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
                for(const t of layer.outputs)
                {
                    if(net.output_tensors.includes(t.name))
                    {
                        const output = new compass.Parameter(t.name,
                            [t].map((output) =>new compass.Argument(output.name,
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

        this._inputs = layer.inputs.map((t) => new compass.Parameter(t.name, [new compass.Argument(t.name, new compass.TensorType(t.type, t.shape), null)]));
        this._outputs = layer.outputs.map((t) => new compass.Parameter(t.name, [new compass.Argument(t.name, new compass.TensorType(t.type, t.shape), null)]));

    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
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

    constructor(blob) {
        this._blob = blob;

        let shape = [];
        if (Object.prototype.hasOwnProperty.call(blob, 'num') &&
            Object.prototype.hasOwnProperty.call(blob, 'channels') &&
            Object.prototype.hasOwnProperty.call(blob, 'width') &&
            Object.prototype.hasOwnProperty.call(blob, 'height')) {
            if (blob.num != 1) {
                shape.push(blob.num);
            }
            if (blob.channels != 1) {
                shape.push(blob.channels);
            }
            if (blob.height != 1) {
                shape.push(blob.height);
            }
            if (blob.width != 1) {
                shape.push(blob.width);
            }
        }
        else if (Object.prototype.hasOwnProperty.call(blob, 'shape')) {
            shape = blob.shape.dim.map((dim) => dim.toNumber());
        }

        let dataType = '?';
        if (blob.data.length > 0) {
            dataType = 'float32';
            this._data = blob.data;
        }
        else if (blob.double_data.length > 0) {
            dataType = 'float64';
            this._data = blob.double_data;
        }

        this._type = new compass.TensorType(dataType, new compass.TensorShape(shape));
    }

    get kind() {
        return 'Blob';
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._context().state;
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
            return '';
        }
        context.limit = 10000;
        const value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

};

compass.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType;
        this._shape = shape;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return (this.dataType || '?') + this._shape.toString();
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

    static open(context) {
        if (compass.Metadata._metadata) {
            return Promise.resolve(compass.Metadata._metadata);
        }
        return context.request('compass-metadata.json', 'utf-8', null).then((data) => {
            compass.Metadata._metadata = new compass.Metadata(data);
            return compass.Metadata._metadata;
        }).catch(() => {
            compass.Metadata._metadata = new compass.Metadata(null);
            return compass.Metadata._metadata;
        });
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

    constructor(metadata, buffer) {
        const reader = text.Reader.open(buffer);
        const sections = []
        let lines = {};
        for (; ;) {
            const line = reader.read();
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
        for(const k of Object.keys(net))
        {
            net[k]=this.parse_param(net[k]);
        }
        net.layers = [];
        for (const i of sections) {
            net.layers.push(this.parse_layer(i));
        }
        this._net = net;
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
        if (prefix + "_scales" in section) {
            scales = this.parse_param(section[prefix + "_scales"]);
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
    parse_layer(section) {
        const layer = {};
        layer.type = String(section["layer_type"]);
        layer.name = String(section["layer_name"]);
        layer.id = section["layer_id"];
        layer.outputs = this.parse_tensor(section, "layer_top");
        layer.inputs = this.parse_tensor(section, "layer_bottom");
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
                delete layer.param[k];
                delete layer.param[type];
                delete layer.param[offset];
                delete layer.param[s];
                layer.param[k.replace("_shape", "")] = t;
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

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = compass.ModelFactory;
}