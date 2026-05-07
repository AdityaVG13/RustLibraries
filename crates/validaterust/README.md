# ValidateRust

ValidateRust is the Pydantic-style validation crate in this workspace.

Current scope:

- In-memory values for null, bool, int, float, text, list, and object data.
- Required and optional fields.
- Primitive and homogeneous-list type checks.
- String minimum-length and integer range constraints.
- Same-data benchmark against Pydantic.

It is not a full Pydantic replacement yet. Schema derivation, serde integration, defaults, coercion modes, nested object schemas, JSON Schema export, rich path errors, and custom validators are roadmap items.
