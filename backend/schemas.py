from pydantic import AliasChoices, BaseModel, ConfigDict, EmailStr, Field, HttpUrl


class RegisterRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    username: str = Field(
        min_length=2,
        max_length=100,
        validation_alias=AliasChoices("username", "Username", "fullName"),
    )
    email: EmailStr = Field(validation_alias=AliasChoices("email", "Email"))
    password: str = Field(
        min_length=6,
        max_length=128,
        validation_alias=AliasChoices("password", "Password"),
    )


class LoginRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    email: EmailStr = Field(validation_alias=AliasChoices("email", "Email"))
    password: str = Field(validation_alias=AliasChoices("password", "Password"))


class UserResponse(BaseModel):
    userId: int
    username: str
    email: str


class AuthResponse(BaseModel):
    success: bool
    message: str
    user: UserResponse | None = None


class VideoUrlRequest(BaseModel):
    video_url: HttpUrl
    user_id: int | None = None


class TeamConfirmationRequest(BaseModel):
    team_1_name: str = Field(min_length=1, max_length=100)
    team_2_name: str = Field(min_length=1, max_length=100)
